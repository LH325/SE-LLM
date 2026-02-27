from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary

import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    # -------------------------
    # helpers
    # -------------------------
    def _is_main_process(self) -> bool:
        return (not self.args.use_multi_gpu) or (getattr(self.args, "local_rank", 0) == 0)

    def _mk_dir(self, d: str):
        if self._is_main_process():
            os.makedirs(d, exist_ok=True)

    def _to_dev(self, x, dtype=torch.float32):
        return x.to(device=self.device, dtype=dtype)

    def _use_m4_preset_if_needed(self):
        if self.args.data != "m4":
            return
        horizon = M4Meta.horizons_map[self.args.seasonal_patterns]
        self.args.token_len = horizon
        self.args.seq_len = 2 * horizon
        self.args.label_len = horizon
        self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]

    # -------------------------
    # required overrides
    # -------------------------
    def _build_model(self):
        self._use_m4_preset_if_needed()

        # IMPORTANT: args.gpu is the *real mapped gpu id* from run.py
        self.device = torch.device(f"cuda:{self.args.gpu}")

        net = self.model_dict[self.args.model].Model(self.args).to(self.device)

        if self.args.use_multi_gpu:
            # DDP wrap (assumes dist.init_process_group already done in run.py)
            net = DDP(net, device_ids=[self.args.gpu], output_device=self.args.gpu)

        return net

    def _get_data(self, flag):
        ds, loader = data_provider(self.args, flag)
        return ds, loader

    def _select_optimizer(self):
        train_params = []
        for pname, p in self.model.named_parameters():
            if p.requires_grad:
                train_params.append(p)
                if self._is_main_process():
                    print(pname, p.dtype, p.shape)

        opt = optim.Adam(
            [{'params': train_params}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        if self._is_main_process():
            print(f"next learning rate is {self.args.learning_rate}")
        return opt

    def _select_criterion(self, name="MSE"):
        key = (name or "MSE").upper()
        if key == "MSE":
            return nn.MSELoss()
        if key == "MAPE":
            return mape_loss()
        if key == "MASE":
            return mase_loss()
        if key == "SMAPE":
            return smape_loss()
        raise ValueError(f"Unknown loss: {name}")

    # -------------------------
    # train / vali / test
    # -------------------------
    def train(self, setting):
        _, tr_loader = self._get_data(flag="train")
        _, va_loader = self._get_data(flag="val")

        save_dir = os.path.join(self.args.checkpoints, setting)
        self._mk_dir(save_dir)

        steps_per_epoch = len(tr_loader)
        stopper = EarlyStopping(self.args, verbose=True)

        optimizer = self._select_optimizer()
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.tmax, eta_min=1e-8
        )
        loss_fn = self._select_criterion(self.args.loss)

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        t_anchor = time.time()

        for ep in range(self.args.train_epochs):
            self.model.train()
            ep_t0 = time.time()
            tick = 0
            loss_hist = []

            for step, (x, y, x_mark, y_mark) in enumerate(tr_loader):
                tick += 1
                optimizer.zero_grad(set_to_none=True)

                # NOTE: short-term model only uses batch_x; other inputs are None
                insample = self._to_dev(x, torch.float32)
                target = self._to_dev(y, torch.float32)
                target_mark = self._to_dev(y_mark, torch.float32)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        pred_seq = self.model(insample, None, None, None)
                        loss = loss_fn(insample, self.args.frequency_map, pred_seq, target, target_mark)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_seq = self.model(insample, None, None, None)
                    loss = loss_fn(insample, self.args.frequency_map, pred_seq, target, target_mark)
                    loss.backward()
                    optimizer.step()

                loss_hist.append(loss.item())

                if (step + 1) % 100 == 0 and self._is_main_process():
                    print(f"\titers: {step + 1}, epoch: {ep + 1} | loss: {loss.item():.7f}")
                    sec_per_iter = (time.time() - t_anchor) / tick
                    eta = sec_per_iter * ((self.args.train_epochs - ep) * steps_per_epoch - step)
                    print(f"\tspeed: {sec_per_iter:.4f}s/iter; left time: {eta:.4f}s")
                    tick = 0
                    t_anchor = time.time()

            if self._is_main_process():
                print(f"Epoch: {ep + 1} cost time: {time.time() - ep_t0}")

            avg_train = float(np.average(loss_hist))

            avg_val = self.vali(tr_loader, va_loader, loss_fn)
            avg_test = avg_val  # keep original behavior

            if self._is_main_process():
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        ep + 1, steps_per_epoch, avg_train, avg_val, avg_test
                    )
                )

            stopper(avg_val, self.model, save_dir)
            if stopper.early_stop:
                if self._is_main_process():
                    print("Early stopping")
                break

            if self.args.cosine:
                lr_sched.step()
                if self._is_main_process():
                    print("lr = {:.10f}".format(optimizer.param_groups[0]["lr"]))
            else:
                adjust_learning_rate(optimizer, ep + 1, self.args)

        ckpt = os.path.join(save_dir, "checkpoint.pth")

        # safer load (cpu -> current device)
        state = torch.load(ckpt, map_location="cpu")
        if self.args.use_multi_gpu:
            # if saved from DDP, may include "module." prefix; keep strict=False anyway
            state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        return self.model

    def vali(self, train_loader, vali_loader, loss_fn):
        insample_np, _ = train_loader.dataset.last_insample_window()
        gt_series = vali_loader.dataset.timeseries

        insample = torch.tensor(insample_np, dtype=torch.float32, device=self.device).unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            B, _, C = insample.shape

            # output buffer on CPU (same as your logic)
            out_buf = torch.zeros((B, self.args.seq_len, C), dtype=torch.float32)
            cut_points = np.append(np.arange(0, B, 500), B)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    for s, e in zip(cut_points[:-1], cut_points[1:]):
                        out_buf[s:e] = self.model(insample[s:e], None, None, None).detach().cpu()
            else:
                for s, e in zip(cut_points[:-1], cut_points[1:]):
                    out_buf[s:e] = self.model(insample[s:e], None, None, None).detach().cpu()

            pred = out_buf[:, -self.args.token_len:, :]  # CPU
            true = torch.from_numpy(np.array(gt_series))  # CPU
            mark = torch.ones(true.shape)  # CPU

            loss = loss_fn(
                insample.detach().cpu()[:, :, 0],
                self.args.frequency_map,
                pred[:, :, 0],
                true,
                mark
            )

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, tr_loader = self._get_data(flag="train")
        _, te_loader = self._get_data(flag="test")

        insample_np, _ = tr_loader.dataset.last_insample_window()
        gt_series = te_loader.dataset.timeseries
        insample = torch.tensor(insample_np, dtype=torch.float32, device=self.device).unsqueeze(-1)

        if test:
            if self._is_main_process():
                print("loading model")

            setting = self.args.test_dir
            ckpt_name = self.args.test_file_name
            ckpt_path = os.path.join(self.args.checkpoints, setting, ckpt_name)

            if self._is_main_process():
                print(f"loading model from {ckpt_path}")

            state = torch.load(ckpt_path, map_location="cpu")
            state = {k.replace("module.", ""): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=False)

        vis_dir = os.path.join("./test_results", setting)
        self._mk_dir(vis_dir)

        self.model.eval()
        with torch.no_grad():
            B, _, C = insample.shape

            # allocate on GPU (same as your logic)
            pred_buf = torch.zeros((B, self.args.seq_len, C), dtype=torch.float32, device=self.device)

            # original behavior: step=1 per sample
            idxs = np.append(np.arange(0, B, 1), B)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    for s, e in zip(idxs[:-1], idxs[1:]):
                        pred_buf[s:e] = self.model(insample[s:e], None, None, None)
            else:
                for s, e in zip(idxs[:-1], idxs[1:]):
                    pred_buf[s:e] = self.model(insample[s:e], None, None, None)

            pred_buf = pred_buf[:, -self.args.token_len:, :]
            preds = pred_buf.detach().cpu().numpy()
            trues = gt_series
            insample_cpu = insample.detach().cpu().numpy()

            # visualization (only main process)
            if self.args.visualize and self._is_main_process():
                for i in range(min(B, 10)):
                    if i % 2 != 0:
                        continue
                    gt = np.concatenate((insample_cpu[i, :, 0], trues[i]), axis=0)
                    pd_line = np.concatenate((insample_cpu[i, :, 0], preds[i, :, 0]), axis=0)
                    visual(gt, pd_line, os.path.join(vis_dir, f"{i}.pdf"))

        if self._is_main_process():
            print("test shape:", preds.shape)

        # -------------------------
        # save forecast csv (main process only)
        # -------------------------
        out_dir = os.path.join("./m4_results", self.args.model)
        self._mk_dir(out_dir)

        cols = [f"V{k + 1}" for k in range(self.args.token_len)]
        df = pd.DataFrame(preds[:, :, 0], columns=cols)
        df.index = te_loader.dataset.ids[:preds.shape[0]]
        df.index.name = "id"

        # keep original behavior (even though it's questionable)
        df.set_index(df.columns[0], inplace=True)

        csv_path = os.path.join(out_dir, f"{self.args.seasonal_patterns}_forecast.csv")
        if self._is_main_process():
            df.to_csv(csv_path)
            print(self.args.model)

        # -------------------------
        # M4 evaluation (main process only)
        # -------------------------
        if self._is_main_process():
            required = {
                "Weekly_forecast.csv",
                "Monthly_forecast.csv",
                "Yearly_forecast.csv",
                "Daily_forecast.csv",
                "Hourly_forecast.csv",
                "Quarterly_forecast.csv",
            }
            existing = set(os.listdir(out_dir))

            if required.issubset(existing):
                evaluator = M4Summary(out_dir, self.args.root_path)
                smape_results, owa_results, mape, mase = evaluator.evaluate()
                print("smape:", smape_results)
                print("mape:", mape)
                print("mase:", mase)
                print("owa:", owa_results)
            else:
                print("After all 6 tasks are finished, you can calculate the averaged index")

        return
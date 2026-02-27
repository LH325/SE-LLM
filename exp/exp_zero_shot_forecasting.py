from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import zero_shot_smape_loss
from utils.m4_summary import M4Summary

import os
import time
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

warnings.filterwarnings("ignore")


def smape_np(y_hat, y):
    return np.mean(200.0 * np.abs(y_hat - y) / (np.abs(y_hat) + np.abs(y) + 1e-8))


def mape_np(y_hat, y):
    return np.mean(np.abs(100.0 * (y_hat - y) / (y + 1e-8)))


class Exp_Zero_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    # -------------------------
    # utilities
    # -------------------------
    def _is_main_process(self) -> bool:
        return (not self.args.use_multi_gpu) or (getattr(self.args, "local_rank", 0) == 0)

    def _mkdir(self, p: str):
        if self._is_main_process():
            os.makedirs(p, exist_ok=True)

    def _to_dev(self, t):
        return t.float().to(self.device)

    def _maybe_set_freq_map(self):
        if self.args.data == "tsf":
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]

    def _forward_core(self, x):
        # model signature: model(batch_x, None, None, None)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(x, None, None, None)
        return self.model(x, None, None, None)

    def _safe_load_state(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")
        # handle DDP prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
        return state

    # -------------------------
    # required overrides
    # -------------------------
    def _build_model(self):
        self._maybe_set_freq_map()

        # IMPORTANT: args.gpu is the real mapped cuda id from run.py (devices mapping)
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
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            train_params.append(p)
            if self._is_main_process():
                print(name, p.dtype, p.shape)

        opt = optim.Adam(
            [{'params': train_params}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        if self._is_main_process():
            print(f"next learning rate is {self.args.learning_rate}")
        return opt

    def _select_criterion(self, loss_name="MSE"):
        tag = (loss_name or "MSE").upper()
        if tag == "MSE":
            return nn.MSELoss()
        if tag == "SMAPE":
            return zero_shot_smape_loss()
        raise ValueError(f"Unknown loss: {loss_name}")

    # -------------------------
    # train / eval
    # -------------------------
    def train(self, setting):
        _, loader_tr = self._get_data(flag="train")
        _, loader_va = self._get_data(flag="val")
        _, loader_te = self._get_data(flag="test")

        self.args.data_path = self.args.test_data_path
        _, loader_te2 = self._get_data(flag="test")

        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        self._mkdir(ckpt_dir)

        steps_per_epoch = len(loader_tr)
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

            per_batch_losses = []
            tick = 0

            for it, (bx, by, bxm, bym) in enumerate(loader_tr):
                tick += 1
                optimizer.zero_grad(set_to_none=True)

                x = self._to_dev(bx)
                y = self._to_dev(by)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat = self.model(x, None, None, None)
                        loss = loss_fn(y_hat, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    y_hat = self.model(x, None, None, None)
                    loss = loss_fn(y_hat, y)
                    loss.backward()
                    optimizer.step()

                per_batch_losses.append(loss.item())

                if (it + 1) % 100 == 0 and self._is_main_process():
                    print(f"\titers: {it + 1}, epoch: {ep + 1} | loss: {loss.item():.7f}")
                    sec_per_iter = (time.time() - t_anchor) / tick
                    eta = sec_per_iter * ((self.args.train_epochs - ep) * steps_per_epoch - it)
                    print(f"\tspeed: {sec_per_iter:.4f}s/iter; left time: {eta:.4f}s")
                    tick = 0
                    t_anchor = time.time()

            if self._is_main_process():
                print(f"Epoch: {ep + 1} cost time: {time.time() - ep_t0}")

            avg_tr = float(np.average(per_batch_losses))
            avg_va = self.vali(None, loader_va, loss_fn)
            avg_te = self.vali2(None, loader_te, loss_fn)
            avg_te2 = self.vali2(None, loader_te2, loss_fn)

            if self._is_main_process():
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} "
                    "Test Loss: {4:.7f} Test Loss2: {5:.7f}".format(
                        ep + 1, steps_per_epoch, avg_tr, avg_va, avg_te, avg_te2
                    )
                )

            stopper(avg_te2, self.model, ckpt_dir)
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

        best_ckpt = os.path.join(ckpt_dir, "checkpoint.pth")
        if self.args.use_multi_gpu:
            dist.barrier()

        state = self._safe_load_state(best_ckpt)
        self.model.load_state_dict(state, strict=False)
        return self.model

    def vali(self, vali_data, vali_loader, loss_fn):
        loss_bucket = []

        self.model.eval()
        with torch.no_grad():
            for bx, by, bxm, bym in vali_loader:
                x = self._to_dev(bx)
                y = by.float()  # keep CPU like original (then detach cpu)
                y_hat = self._forward_core(x)

                pred_cpu = y_hat.detach().cpu()
                true_cpu = y.detach().cpu()
                loss_bucket.append(loss_fn(pred_cpu, true_cpu))

        avg_loss = float(np.average(loss_bucket))
        self.model.train()
        return avg_loss

    def _rolling_forecast(self, context_x):

        steps = self.args.test_pred_len // self.args.token_len
        rem = self.args.test_pred_len - steps * self.args.token_len
        if rem != 0:
            steps += 1

        chunks = []
        x = context_x

        for _ in range(steps):
            if chunks:
                x = torch.cat([x[:, self.args.token_len:, :], chunks[-1]], dim=1)
            out = self._forward_core(x)
            chunks.append(out[:, -self.args.token_len:, :])

        pred_seq = torch.cat(chunks, dim=1)
        if rem != 0:
            pred_seq = pred_seq[:, :-rem, :]

        return pred_seq

    def vali2(self, vali_data, vali_loader, loss_fn):
        losses = []
        weights = []

        self.model.eval()
        with torch.no_grad():
            for bx, by, bxm, bym in vali_loader:
                x = self._to_dev(bx)
                y_full = by.float()

                y_hat = self._rolling_forecast(x)
                y_true = y_full[:, -self.args.test_pred_len:, :].to(self.device)

                pred_cpu = y_hat.detach().cpu()
                true_cpu = y_true.detach().cpu()

                losses.append(loss_fn(pred_cpu, true_cpu))
                weights.append(x.shape[0])

        avg_loss = float(np.average(losses, weights=weights))
        self.model.train()
        return avg_loss

    def test(self, setting, test=0):
        if test:
            if self._is_main_process():
                print("loading model")

            setting = self.args.test_dir
            ckpt_name = self.args.test_file_name
            ckpt_path = os.path.join(self.args.checkpoints, setting, ckpt_name)

            if self._is_main_process():
                print(f"loading model from {ckpt_path}")

            state = self._safe_load_state(ckpt_path)
            self.model.load_state_dict(state, strict=False)

        # switch to test data path and rebuild loader (same as original)
        self.args.data_path = self.args.test_data_path
        _, loader_te = self._get_data("test")

        all_preds, all_trues = [], []

        self.model.eval()
        with torch.no_grad():
            for bx, by, bxm, bym in loader_te:
                x = self._to_dev(bx)
                y_full = by.float()

                y_hat = self._rolling_forecast(x)
                y_true = y_full[:, -self.args.test_pred_len:, :].to(self.device)

                all_preds.append(y_hat.detach().cpu().numpy())
                all_trues.append(y_true.detach().cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)

        if self._is_main_process():
            print("test shape:", preds.shape, trues.shape)
            s = smape_np(preds, trues)
            m = mape_np(preds, trues)
            print("mape:{:4f}, smape:{:.4f}".format(m, s))

        return
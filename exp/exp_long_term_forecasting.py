from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    # -------------------------
    # helpers
    # -------------------------
    def _is_main_process(self) -> bool:
        return (not self.args.use_multi_gpu) or (getattr(self.args, "local_rank", 0) == 0)

    def _ensure_dir(self, folder: str):
        if self._is_main_process():
            os.makedirs(folder, exist_ok=True)

    def _move_batch(self, x, y, x_mark, y_mark):
        x = x.float().to(self.device)
        y = y.float()  # keep on CPU first; later move to device based on usage
        x_mark = x_mark.float().to(self.device)
        y_mark = y_mark.float().to(self.device)
        return x, y, x_mark, y_mark

    # -------------------------
    # core methods
    # -------------------------
    def _build_model(self):
        net = self.model_dict[self.args.model].Model(self.args)

        # IMPORTANT:
        # In the new run.py, args.gpu is the *real* mapped CUDA id (e.g., 2 or 3),
        # while local_rank is just the process index (0/1/...).
        self.device = torch.device(f"cuda:{self.args.gpu}")

        if self.args.use_multi_gpu:
            net = net.to(self.device)
            net = DDP(net, device_ids=[self.args.gpu], output_device=self.args.gpu)
        else:
            net = net.to(self.device)

        return net

    def _get_data(self, flag):
        ds, loader = data_provider(self.args, flag)
        return ds, loader

    def _select_optimizer(self):
        trainable_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            trainable_params.append(param)
            if self._is_main_process():
                print(name, param.dtype, param.shape)

        opt = optim.Adam(
            [{'params': trainable_params}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        if self._is_main_process():
            print(f"next learning rate is {self.args.learning_rate}")
        return opt

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, eval_data, eval_loader, loss_fn, is_test=False):
        batch_losses = []
        batch_sizes = []

        self.model.eval()
        t0 = time.time()
        n_steps = len(eval_loader)
        tick = 0

        with torch.no_grad():
            for step_idx, (x, y, x_mark, y_mark) in enumerate(eval_loader):
                tick += 1
                x, y, x_mark, y_mark = self._move_batch(x, y, x_mark, y_mark)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat = self.model(x, x_mark, None, y_mark)
                else:
                    y_hat = self.model(x, x_mark, None, y_mark)

                if is_test:
                    y_hat = y_hat[:, -self.args.token_len:, :]
                    y = y[:, -self.args.token_len:, :].to(self.device)
                else:
                    y_hat = y_hat[:, :, :]
                    y = y[:, :, :].to(self.device)

                loss_val = loss_fn(y_hat, y).detach().cpu()
                batch_losses.append(loss_val)
                batch_sizes.append(x.shape[0])

                if (step_idx + 1) % 100 == 0 and self._is_main_process():
                    sec_per_iter = (time.time() - t0) / tick
                    eta = sec_per_iter * (n_steps - step_idx)
                    print(f"\titers: {step_idx + 1}, speed: {sec_per_iter:.4f}s/iter, left time: {eta:.4f}s")
                    tick = 0
                    t0 = time.time()

        if self.args.use_multi_gpu:
            local_avg = torch.tensor(
                np.average(batch_losses, weights=batch_sizes),
                device=self.device
            )
            dist.barrier()
            dist.all_reduce(local_avg, op=dist.ReduceOp.SUM)
            final_loss = local_avg.item() / dist.get_world_size()
        else:
            final_loss = float(np.average(batch_losses, weights=batch_sizes))

        self.model.train()
        return final_loss

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, val_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        self._ensure_dir(ckpt_dir)

        n_train_steps = len(train_loader)
        stopper = EarlyStopping(self.args, verbose=True)

        optimizer = self._select_optimizer()
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.tmax, eta_min=1e-8
        )
        loss_fn = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        global_t0 = time.time()

        for ep in range(self.args.train_epochs):
            self.model.train()
            ep_t0 = time.time()
            tick = 0

            sum_loss = torch.tensor(0.0, device=self.device)
            n_batches = torch.tensor(0.0, device=self.device)

            for step, (x, y, x_mark, y_mark) in enumerate(train_loader):
                tick += 1
                optimizer.zero_grad(set_to_none=True)

                x = x.float().to(self.device)
                y = y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat = self.model(x, x_mark, None, y_mark)
                        loss = loss_fn(y_hat, y)

                    sum_loss += loss
                    n_batches += 1

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    y_hat = self.model(x, x_mark, None, y_mark)
                    loss = loss_fn(y_hat, y)

                    sum_loss += loss
                    n_batches += 1

                    loss.backward()
                    optimizer.step()

                if (step + 1) % 100 == 0 and self._is_main_process():
                    print(f"\titers: {step + 1}, epoch: {ep + 1} | loss: {loss.item():.7f}")
                    sec_per_iter = (time.time() - global_t0) / tick
                    eta = sec_per_iter * ((self.args.train_epochs - ep) * n_train_steps - step)
                    print(f"\tspeed: {sec_per_iter:.4f}s/iter; left time: {eta:.4f}s")
                    tick = 0
                    global_t0 = time.time()

            if self._is_main_process():
                print(f"Epoch: {ep + 1} cost time: {time.time() - ep_t0}")

            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)

            train_loss = sum_loss.item() / max(n_batches.item(), 1.0)

            val_loss = self.vali(None, val_loader, loss_fn, is_test=False)
            tst_loss = self.vali(None, test_loader, loss_fn, is_test=True)

            if self._is_main_process():
                print(
                    f"Epoch: {ep + 1}, Steps: {n_train_steps} | "
                    f"Train Loss: {train_loss:.7f} Vali Loss: {val_loss:.7f} Test Loss: {tst_loss:.7f}"
                )

            stopper(tst_loss, self.model, ckpt_dir)
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

            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(ep + 1)

        best_ckpt = os.path.join(ckpt_dir, "checkpoint.pth")
        if self.args.use_multi_gpu:
            dist.barrier()

        # safer load
        state = torch.load(best_ckpt, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')

        if self._is_main_process():
            print("info:", self.args.test_seq_len, self.args.test_label_len,
                  self.args.token_len, self.args.test_pred_len)

        if test:
            if self._is_main_process():
                print("loading model")

            setting = self.args.test_dir
            ckpt_name = self.args.test_file_name
            ckpt_path = os.path.join(self.args.checkpoints, setting, ckpt_name)

            if self._is_main_process():
                print(f"loading model from {ckpt_path}")

            state = torch.load(ckpt_path, map_location="cpu")
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            self.model.load_state_dict(cleaned, strict=False)

        pred_list, true_list = [], []

        out_dir = os.path.join("./test_results", setting)
        if self._is_main_process():
            os.makedirs(out_dir, exist_ok=True)

        self.model.eval()
        t0 = time.time()
        n_steps = len(test_loader)
        tick = 0

        with torch.no_grad():
            for idx, (x, y, x_mark, y_mark) in enumerate(test_loader):
                tick += 1
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                full_h = self.args.test_pred_len
                stride = self.args.token_len

                k = full_h // stride
                remain = full_h - k * stride
                if remain != 0:
                    k += 1

                chunks = []
                for j in range(k):
                    if chunks:
                        x = torch.cat([x[:, stride:, :], chunks[-1]], dim=1)
                        tmp_mark = y_mark[:, j - 1:j, :]
                        x_mark = torch.cat([x_mark[:, 1:, :], tmp_mark], dim=1)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            out = self.model(x, x_mark, None, y_mark)
                    else:
                        out = self.model(x, x_mark, None, y_mark)

                    chunks.append(out[:, -stride:, :])

                y_hat = torch.cat(chunks, dim=1)
                if remain != 0:
                    y_hat = y_hat[:, :-(stride - remain), :]

                y_true = y[:, -full_h:, :]

                pred_cpu = y_hat.detach().cpu()
                true_cpu = y_true.detach().cpu()

                pred_list.append(pred_cpu)
                true_list.append(true_cpu)

                if (idx + 1) % 100 == 0 and self._is_main_process():
                    sec_per_iter = (time.time() - t0) / tick
                    eta = sec_per_iter * (n_steps - idx)
                    print(f"\titers: {idx + 1}, speed: {sec_per_iter:.4f}s/iter, left time: {eta:.4f}s")
                    tick = 0
                    t0 = time.time()

                # visualize only on main process to avoid duplicated files
                if self._is_main_process():
                    gt = np.array(true_cpu[0, :, -1])
                    pd = np.array(pred_cpu[0, :, -1])
                    lookback = x[0, :, -1].detach().cpu().numpy()

                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)

                    vis_dir = os.path.join(out_dir, f"{self.args.test_pred_len}")
                    os.makedirs(vis_dir, exist_ok=True)
                    visual(gt, pd, os.path.join(vis_dir, f"{idx}.png"))

        preds = torch.cat(pred_list, dim=0).numpy()
        trues = torch.cat(true_list, dim=0).numpy()

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        if self._is_main_process():
            print(f"mse:{mse}, mae:{mae}")
            with open("result_long_term_forecast.txt", "a") as fp:
                fp.write(setting + "  \n")
                fp.write(f"mse:{mse}, mae:{mae}\n\n")

        return
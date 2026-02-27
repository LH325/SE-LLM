import math
from datetime import datetime
from distutils.util import strtobool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

plt.switch_backend("agg")


def adjust_learning_rate(opt, cur_epoch, cfg):
    """
    Adjust learning rate given current epoch and cfg.lradj.
    """
    schedule = None
    if cfg.lradj == "type1":
        schedule = {cur_epoch: cfg.learning_rate * (0.5 ** (cur_epoch - 1))}
    elif cfg.lradj == "type2":
        schedule = {cur_epoch: cfg.learning_rate * (0.6 ** cur_epoch)}
    elif cfg.lradj == "cosine":
        schedule = {
            cur_epoch: (cfg.learning_rate / 2.0) * (1.0 + math.cos(cur_epoch / cfg.train_epochs * math.pi))
        }

    if schedule is None:
        return

    if cur_epoch in schedule:
        new_lr = schedule[cur_epoch]
        for group in opt.param_groups:
            group["lr"] = new_lr

        if (cfg.use_multi_gpu and cfg.local_rank == 0) or (not cfg.use_multi_gpu):
            print("next learning rate is {}".format(new_lr))


class EarlyStopping:
    def __init__(self, cfg, verbose=False, delta=0.0):
        self.patience = cfg.patience
        self.verbose = verbose
        self.delta = delta

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_loss = np.Inf

        self._ddp = bool(cfg.use_multi_gpu)
        self._rank0 = (getattr(cfg, "local_rank", 0) == 0) if self._ddp else True

    def __call__(self, val_loss, net, save_dir):
        score = -val_loss

        # first time
        if self.best_score is None:
            self.best_score = score
            if self.verbose and self._rank0:
                print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).")
            self.best_loss = val_loss
            self._save_if_needed(val_loss, net, save_dir)
            return

        # no improvement
        if score < self.best_score + self.delta:
            self.counter += 1
            if self._rank0:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        # improvement
        self.best_score = score
        self._save_if_needed(val_loss, net, save_dir)
        if self.verbose and self._rank0:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).")
        self.best_loss = val_loss
        self.counter = 0

    def _save_if_needed(self, val_loss, net, save_dir):
        if self._ddp:
            if self._rank0:
                self.save_checkpoint(val_loss, net, save_dir)
            dist.barrier()
        else:
            self.save_checkpoint(val_loss, net, save_dir)

    def save_checkpoint(self, val_loss, net, save_dir):
        # only keep parameters with requires_grad=True (same behavior as your original)
        grad_flag = {name: p.requires_grad for (name, p) in net.named_parameters()}
        state = net.state_dict()

        # delete frozen params
        for key in list(state.keys()):
            if key in grad_flag and (not grad_flag[key]):
                del state[key]

        ckpt_path = save_dir + "/" + "checkpoint.pth"
        torch.save(state, ckpt_path)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, arr):
        return (arr - self.mean) / self.std

    def inverse_transform(self, arr):
        return (arr * self.std) + self.mean


def visual(gt, pred=None, name="./pic/test.pdf"):
    plt.figure()
    plt.plot(gt, label="GroundTruth", linewidth=2)
    if pred is not None:
        plt.plot(pred, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    """
    Post-process anomaly predictions to cover contiguous anomaly segments.
    """
    in_anomaly = False
    for idx in range(len(gt)):
        if gt[idx] == 1 and pred[idx] == 1 and (not in_anomaly):
            in_anomaly = True

            # backward fill
            for j in range(idx, 0, -1):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1

            # forward fill
            for j in range(idx, len(gt)):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1

        elif gt[idx] == 0:
            in_anomaly = False

        if in_anomaly:
            pred[idx] = 1

    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def convert_tsf_to_dataframe(
    tsf_path,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    """
    Convert a .tsf file to a Pandas DataFrame.

    Returns:
        (df, frequency, horizon, contain_missing, contain_equal_length)
    """
    attr_names = []
    attr_types = []
    meta_store = {}

    n_lines = 0
    freq = None
    horizon = None
    has_missing = None
    is_equal_len = None

    seen_data_tag = False
    seen_data_section = False
    reading_series = False

    with open(tsf_path, "r", encoding="cp1252") as fp:
        for raw in fp:
            line = raw.strip()

            if not line:
                n_lines += 1
                continue

            if line.startswith("@"):
                if line.startswith("@data"):
                    if len(attr_names) == 0:
                        raise Exception("Missing attribute section. Attribute section must come before data.")
                    seen_data_tag = True
                    n_lines += 1
                    continue

                # meta line (not @data)
                parts = line.split(" ")
                if line.startswith("@attribute"):
                    if len(parts) != 3:
                        raise Exception("Invalid meta-data specification.")
                    attr_names.append(parts[1])
                    attr_types.append(parts[2])
                else:
                    if len(parts) != 2:
                        raise Exception("Invalid meta-data specification.")
                    tag, val = parts[0], parts[1]
                    if tag == "@frequency":
                        freq = val
                    elif tag == "@horizon":
                        horizon = int(val)
                    elif tag == "@missing":
                        has_missing = bool(strtobool(val))
                    elif tag == "@equallength":
                        is_equal_len = bool(strtobool(val))
                n_lines += 1
                continue

            if line.startswith("#"):
                n_lines += 1
                continue

            # data line
            if len(attr_names) == 0:
                raise Exception("Missing attribute section. Attribute section must come before data.")
            if not seen_data_tag:
                raise Exception("Missing @data tag.")

            if not reading_series:
                reading_series = True
                seen_data_section = True
                series_list = []
                for col in attr_names:
                    meta_store[col] = []

            full_parts = line.split(":")
            if len(full_parts) != (len(attr_names) + 1):
                raise Exception("Missing attributes/values in series.")

            raw_series = full_parts[-1].split(",")
            if len(raw_series) == 0:
                raise Exception(
                    "A given series should contains a set of comma separated numeric values. "
                    "At least one numeric value should be there in a series. "
                    "Missing values should be indicated with ? symbol"
                )

            parsed_vals = []
            for v in raw_series:
                if v == "?":
                    parsed_vals.append(replace_missing_vals_with)
                else:
                    parsed_vals.append(float(v))

            if parsed_vals.count(replace_missing_vals_with) == len(parsed_vals):
                raise Exception(
                    "All series values are missing. A given series should contains a set of comma separated numeric values. "
                    "At least one numeric value should be there in a series."
                )

            series_list.append(pd.Series(parsed_vals).array)

            for i in range(len(attr_names)):
                typ = attr_types[i]
                raw_val = full_parts[i]

                if typ == "numeric":
                    att_val = int(raw_val)
                elif typ == "string":
                    att_val = str(raw_val)
                elif typ == "date":
                    att_val = datetime.strptime(raw_val, "%Y-%m-%d %H-%M-%S")
                else:
                    raise Exception("Invalid attribute type.")

                meta_store[attr_names[i]].append(att_val)

            n_lines += 1

    if n_lines == 0:
        raise Exception("Empty file.")
    if len(attr_names) == 0:
        raise Exception("Missing attribute section.")
    if not seen_data_section:
        raise Exception("Missing series information under data section.")

    meta_store[value_column_name] = series_list
    df = pd.DataFrame(meta_store)

    return df, freq, horizon, has_missing, is_equal_len
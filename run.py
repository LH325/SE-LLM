import argparse
import os

import torch
import torch.distributed as dist

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
import numpy as np
import random
def is_main_process(args) -> bool:
    return (not args.use_multi_gpu) or (getattr(args, "local_rank", 0) == 0)


def _parse_devices(devices: str):
    """
    devices: "0" or "0,1,2"
    returns: [0] or [0,1,2]
    """
    if devices is None:
        return None
    s = devices.strip()
    if not s:
        return None
    try:
        ids = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    except ValueError as e:
        raise ValueError(f"--devices must be like '0' or '0,1'. Got: {devices}") from e
    if len(ids) == 0:
        return None
    # de-dup while keeping order
    seen = set()
    uniq = []
    for i in ids:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SELLM")

    parser.add_argument("--task_name", type=str, default="long_term_forecast",
                        choices=["long_term_forecast", "short_term_forecast", "zero_shot_forecast"])
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--itr", type=int, default=1)

    parser.add_argument("--model_id", type=str, default="Weather_Qwen2")
    parser.add_argument("--model", type=str, default="SELLM")

    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default=r"D:\TimeSeriesForecasting\AutoTimes\Timeseriesdata")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--test_data_path", type=str, default="ETTh1.csv")

    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--test_file_name", type=str, default="checkpoint.pth")

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=384)
    parser.add_argument("--drop_last", action="store_true", default=False)
    parser.add_argument("--val_set_shuffle", action="store_false", default=True)
    parser.add_argument("--drop_short", action="store_true", default=False)

    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--label_len", type=int, default=576)
    parser.add_argument("--token_len", type=int, default=96)
    parser.add_argument("--test_seq_len", type=int, default=672)
    parser.add_argument("--test_label_len", type=int, default=576)
    parser.add_argument("--test_pred_len", type=int, default=96)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--word_size", type=int, default=1500)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--llm_ckp_dir", type=str, default="Qwen/Qwen-0.5B-GRPO")
    parser.add_argument("--mlp_hidden_dim", type=int, default=1024)
    parser.add_argument("--mlp_activation", type=str, default="relu")

    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type2")
    parser.add_argument("--tmax", type=int, default=10)
    parser.add_argument("--des", type=str, default="Exp")

    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--cosine", action="store_true", default=True)

    # single GPU
    parser.add_argument("--gpu", type=int, default=0)

    # multi GPU (DDP)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="",
                        help="Comma-separated CUDA device ids for DDP, e.g. '0' or '0,1'. "
                             "Use with torchrun --nproc_per_node=<len(devices)> and --use_multi_gpu.")
    parser.add_argument("--visualize", action="store_true", default=False)

    return parser


def setup_distributed(args):
    """
    DDP with specified GPUs:
    - Example: --devices "0,1"
      torchrun --nproc_per_node=2 run.py --use_multi_gpu --devices "0,1" ...
    Each process uses CUDA device devices[LOCAL_RANK].
    """
    args.device_ids = _parse_devices(args.devices)  # list[int] or None

    if not args.use_multi_gpu:
        # single process
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1

        # choose GPU: if devices is provided, use its first one; else use --gpu
        if args.device_ids and len(args.device_ids) > 0:
            args.gpu = args.device_ids[0]
        return

    # DDP mode: must be launched by torchrun
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not args.device_ids:
        raise ValueError("In --use_multi_gpu mode, you must provide --devices like '0,1'.")

    if args.world_size != len(args.device_ids):
        # avoid silent mismatch/hang
        raise RuntimeError(
            f"DDP world_size({args.world_size}) != len(devices)({len(args.device_ids)}). "
            f"Use torchrun --nproc_per_node={len(args.device_ids)} and --devices '{args.devices}'."
        )

    n_visible = torch.cuda.device_count()
    if n_visible <= 0:
        raise RuntimeError("No CUDA device found, but --use_multi_gpu is set.")

    if args.local_rank < 0 or args.local_rank >= len(args.device_ids):
        raise RuntimeError(f"LOCAL_RANK={args.local_rank} out of range for devices={args.device_ids}")

    mapped_gpu = args.device_ids[args.local_rank]
    if mapped_gpu < 0 or mapped_gpu >= n_visible:
        raise RuntimeError(f"Requested GPU id {mapped_gpu} but cuda device_count={n_visible}")

    args.gpu = mapped_gpu  # IMPORTANT: let Exp use this
    torch.cuda.set_device(mapped_gpu)

    dist.init_process_group(backend="nccl")

    if is_main_process(args):
        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")
        print(f"[DDP] world_size={args.world_size}, rank={args.rank}, local_rank={args.local_rank}, "
              f"devices={args.device_ids}, mapped_gpu={mapped_gpu}, master={master_addr}:{master_port}, "
              f"cuda_device_count={n_visible}")


def pick_experiment(task_name: str):
    mapping = {
        "long_term_forecast": Exp_Long_Term_Forecast,
        "short_term_forecast": Exp_Short_Term_Forecast,
        "zero_shot_forecast": Exp_Zero_Shot_Forecast,
    }
    return mapping.get(task_name, Exp_Long_Term_Forecast)


def make_setting(args, run_id: int) -> str:
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}"
        f"_sl{args.seq_len}_ll{args.label_len}_tl{args.token_len}"
        f"_lr{args.learning_rate}_bt{args.batch_size}_wd{args.weight_decay}"
        f"_cos{args.cosine}_{args.des}_{run_id}"
    )


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    parser = build_parser()
    args = parser.parse_args()

    setup_distributed(args)

    ExpCls = pick_experiment(args.task_name)

    if args.is_training:
        for ii in range(args.itr):
            exp = ExpCls(args)
            setting = make_setting(args, ii)

            if is_main_process(args):
                print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            if is_main_process(args):
                print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = make_setting(args, ii)
        exp = ExpCls(args)
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

    if args.use_multi_gpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
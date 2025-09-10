import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from contextlib import contextmanager
import re

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def create_logger(cfg, cfg_path, phase='train', rank=-1):
    # set up logger dir
    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_path = os.path.basename(cfg_path).split('.')[0]

    if rank in [-1, 0]:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = '{}_{}_{}.log'.format(cfg_path, time_str, phase)
        # set up tensorboard_log_dir
        tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                                  (cfg_path + '_' + time_str)
        final_output_dir = tensorboard_log_dir
        if not tensorboard_log_dir.exists():
            print('=> creating {}'.format(tensorboard_log_dir))
            tensorboard_log_dir.mkdir(parents=True)

        final_log_file = tensorboard_log_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger, str(final_output_dir), str(tensorboard_log_dir)
    else:
        return None, None, None


def select_device(logger=None, device: str = "") -> torch.device:
    """
    Cross-platform device picker.
    Accepts: "", "auto", "cpu", "mps", "cuda", "cuda:0", "0", "0,1", etc.
    Prefers CUDA (if explicitly requested or available), then MPS, then CPU.
    Never asserts on CUDA when unavailable; falls back gracefully.
    """
    d = (device or "").strip().lower()

    def log(msg):
        if logger is not None:
            try:
                logger.info(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

    # Helper: MPS availability (Apple Silicon)
    def mps_ok():
        try:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except Exception:
            return False

    # Normalize some aliases
    if d in ("", "auto", "default"):
        # Auto-pick: prefer CUDA, then MPS, else CPU
        if torch.cuda.is_available():
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # respect user's setup
            log("Auto device: CUDA")
            return torch.device("cuda:0")
        if mps_ok():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""       # ensure no CUDA path is touched
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            log("Auto device: MPS (Apple Silicon)")
            return torch.device("mps")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""           # force CPU path
        log("Auto device: CPU")
        return torch.device("cpu")

    if d in ("cpu",):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""           # disable CUDA
        log("Device: CPU")
        return torch.device("cpu")

    if d in ("mps",):
        if mps_ok():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            log("Device: MPS")
            return torch.device("mps")
        log("Requested MPS but it's unavailable; falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return torch.device("cpu")

    # CUDA-style strings: "cuda", "cuda:0"
    if d.startswith("cuda"):
        if torch.cuda.is_available():
            log(f"Device: {d}")
            return torch.device(d)
        log("Requested CUDA but it's unavailable; falling back to CPU/MPS.")
        if mps_ok():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            log("Fallback: MPS")
            return torch.device("mps")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log("Fallback: CPU")
        return torch.device("cpu")

    # Numeric GPU indices like "0" or "0,1"
    if d.replace(",", "").isdigit():
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = d
            first = d.split(",")[0]
            log(f"Device: cuda:{first} (CUDA_VISIBLE_DEVICES={d})")
            return torch.device(f"cuda:{first}")
        log("Requested CUDA index but CUDA is unavailable; falling back to CPU/MPS.")
        if mps_ok():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            log("Fallback: MPS")
            return torch.device("mps")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log("Fallback: CPU")
        return torch.device("cpu")

    # Last-resort: try to construct a torch.device or default to CPU
    try:
        dev = torch.device(d)
        log(f"Device: {dev}")
        return dev
    except Exception:
        log(f"Unrecognized device '{device}', falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return torch.device("cpu")


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR0,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            #model.parameters(),
            lr=cfg.TRAIN.LR0,
            betas=(cfg.TRAIN.MOMENTUM, 0.999)
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            #model.parameters(),
            lr=cfg.TRAIN.LR0,
            betas=(cfg.TRAIN.MOMENTUM, 0.999)
        )

    return optimizer


def save_checkpoint(epoch, name, model, optimizer, output_dir, filename, is_best=False):
    model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    checkpoint = {
            'epoch': epoch,
            'model': name,
            'state_dict': model_state,
            # 'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in checkpoint:
        torch.save(checkpoint['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
        # elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def meshgrid(*tensors):
        return torch.meshgrid(*tensors)

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

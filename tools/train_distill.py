import argparse, os, time
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

from lib.core.distill import SegDistiller

def build_teacher(ckpt_path: str, device: torch.device, fp16_ckpt: bool = True) -> nn.Module:
    """
    Instantiate YOLOPX model and load seg-only FP16 weights.
    If the architecture still contains a detect head, it's fine: we will be never using use it
    """
    from lib.models.YOLOP import YOLOP as Model
    teacher = Model()
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state)
    msg = teacher.load_state_dict(sd, strict=False)
    print("Loaded teacher:", msg)

    teacher.to(device)
    if fp16_ckpt:
        teacher.half()  # match your FP16 state dict for memory/bandwidth
    teacher.eval()
    return teacher

def build_student(width_mult: float, device: torch.device) -> nn.Module:
    """
    Small student
    Option A: width-multiplied YOLOPX-Seg
    Option B: smaller trunk + two seg heads
    """
    # --- BEGIN: simplest baseline: reuse same model but add width_mult if you have it ---
    from lib.models.YOLOP import YOLOP as Model
    try:
        student = Model(width_mult=width_mult)
    except TypeError:
        student = Model()  # fallback; still works, not smaller though
    # --- END ---
    student.to(device).train()
    return student

def get_dataloaders(args):
    """
    !!! important !!! I stripped the vehicle detections, so we focus on Driving Area and Lane Lines only here
    The batch should yield:
      images: [B,3,H,W] float, 0..1
      targets: {"da": Long[B,H,W] or {0,1}, "ll": Long[B,H,W] or {0,1]}
    """
    raise NotImplementedError("Wire this to your dataset pipeline.")

def set_fast_mode():
    torch.backends.cudnn.benchmark = True

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_fast_mode()

    teacher = build_teacher(args.teacher, device, fp16_ckpt=True)
    student = build_student(args.student_width, device)

    distiller = SegDistiller(teacher, student, T=args.T, alpha=args.alpha, beta=args.beta,
                             ignore_index=args.ignore_index, use_dice=not args.no_dice, dice_weight=args.dice_w).to(device)

    # Optimizer / sched
    opt = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler()

    # Data
    train_loader, val_loader = get_dataloaders(args)

    best_val = 1e9
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        t0 = time.time()
        meters = {"loss": 0.0}
        for i, batch in enumerate(train_loader):
            images = batch["img"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            # Stay in fp32 here; AMP handles casts. Teacher is half() already.
            targets = {"da": batch["da"].to(device, non_blocking=True),
                       "ll": batch["ll"].to(device, non_blocking=True)}

            opt.zero_grad(set_to_none=True)
            with autocast():
                loss, logs = distiller(images, targets)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            meters["loss"] += logs["total"]

        sched.step()
        epoch_loss = meters["loss"] / max(1, len(train_loader))
        print(f"[Epoch {epoch:03d}] train_loss={epoch_loss:.4f}  time={time.time()-t0:.1f}s  lr={sched.get_last_lr()[0]:.2e}")

        val_loss = validate(distiller, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"student": student.state_dict(),
                        "epoch": epoch,
                        "args": vars(args)}, os.path.join(args.out, f"student_best.pt"))
            print(f"  ✓ Saved best student (val_loss={best_val:.4f})")

    print("Done.")

@torch.no_grad()
def validate(distiller: SegDistiller, loader, device):
    distiller.eval()
    loss_sum = 0.0
    for batch in loader:
        images = batch["img"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = {"da": batch["da"].to(device, non_blocking=True),
                   "ll": batch["ll"].to(device, non_blocking=True)}

        loss, _ = distiller(images, targets)
        loss_sum += float(loss.detach().cpu())
    distiller.train()
    return loss_sum / max(1, len(loader))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight for hard loss")
    ap.add_argument("--beta", type=float, default=0.5, help="weight for KD loss")
    ap.add_argument("--ignore_index", type=int, default=255)
    ap.add_argument("--no_dice", action="store_true")
    ap.add_argument("--dice_w", type=float, default=0.2)
    ap.add_argument("--student_width", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="runs/distill")
    args = ap.parse_args()
    run(args)

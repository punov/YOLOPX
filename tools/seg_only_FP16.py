import re, torch
ckpt = torch.load("weights/epoch-195.pth", map_location="cpu")
sd = ckpt.get('state_dict', ckpt)

print("\n".join([k for k in sd if re.search(r"(detect|det_head|yolo\.|cls_pred|reg_pred)", k, re.I)][:50]))

drop = [re.compile(pat, re.I) for pat in [r"detect", r"det_head", r"yolo\.", r"\.cls\.", r"\.reg\."]]
sd_seg = {k: v for k, v in sd.items() if not any(p.search(k) for p in drop)}

# Convert FP32 params to FP16 for disk
for k, v in list(sd_seg.items()):
    if torch.is_floating_point(v) and v.dtype == torch.float32:
        sd_seg[k] = v.half()

torch.save({'state_dict': sd_seg}, "weights/epoch-195-segonly-fp16.pth")
import os
import sys
from PIL import Image, ImageOps, ImageDraw
import json
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
def load_gray(p):
    im = Image.open(p)
    return im.convert("L")
def overlay_mask(frame, mask, color=(255,0,0), alpha=0.4):
    fr = frame.convert("RGB")
    thr = mask.point(lambda v: 255 if v>0 else 0, mode="L")
    col = Image.new("RGB", fr.size, color)
    col.putalpha(thr)
    base = fr.copy()
    base.putalpha(255)
    out = Image.alpha_composite(base, col)
    return out.convert("RGB")
def draw_histogram(img, out_path, width=512, height=256):
    hist = img.histogram()
    bins = hist[:256]
    m = max(bins) if bins else 1
    canvas = Image.new("RGB", (width, height), (255,255,255))
    d = ImageDraw.Draw(canvas)
    bw = width//256
    for i,v in enumerate(bins):
        h = int((v/m)*(height-10))
        x0 = i*bw
        y0 = height-h
        d.rectangle([x0,y0,x0+bw-1,height], fill=(50,100,200))
    canvas.save(out_path)
def main():
    sys.path.append("c:\\DATA CELL LINEAGE REGENERATION")
    from src.data.loader import PhCU373Dataset
    root = os.path.join("c:\\DATA CELL LINEAGE REGENERATION","Datasets","PhC-C2DH-U373")
    outdir = os.path.join("c:\\DATA CELL LINEAGE REGENERATION","outputs","eda")
    ensure_dir(outdir)
    ds = PhCU373Dataset(root)
    frames = ds.list_frames()
    st = ds.list_st_masks()
    targets = []
    if frames:
        targets.append(0)
        mid = len(frames)//2
        targets.append(mid)
        targets.append(len(frames)-1)
    for idx in targets:
        fpath = frames[idx]
        mpath = st[idx] if idx<len(st) else None
        f = load_gray(fpath)
        f.save(os.path.join(outdir, f"frame_{idx:03d}.png"))
        draw_histogram(f, os.path.join(outdir, f"hist_{idx:03d}.png"))
        if mpath:
            m = load_gray(mpath)
            ov = overlay_mask(f, m, (255,0,0), 0.4)
            ov.save(os.path.join(outdir, f"overlay_st_{idx:03d}.png"))
    gtseg = ds.list_gt_seg()
    if gtseg:
        g0 = load_gray(gtseg[0])
        f0 = load_gray(frames[ds.frame_index_from_name(frames[0])])
        ovg = overlay_mask(f0, g0, (0,255,0), 0.4)
        ovg.save(os.path.join(outdir, "overlay_gtseg_sample.png"))
    tracks = ds.parse_tracks()
    divs = [t for t in tracks if t[3]>0]
    if divs:
        cid,start,end,parent = divs[0]
        if start<len(frames):
            f = load_gray(frames[start])
            f.save(os.path.join(outdir, f"division_frame_{start:03d}.png"))
    def parse_idx_from_gt(name):
        b = os.path.basename(name)
        s = "".join(ch for ch in b if ch.isdigit())
        return int(s) if s.isdigit() else None
    def bin_image(img):
        try:
            import numpy as np
            arr = np.array(img)
            return (arr>0)
        except Exception:
            data = list(img.getdata())
            return [1 if v>0 else 0 for v in data], img.size
    def metrics(gt_img, st_img):
        try:
            import numpy as np
            g = (np.array(gt_img)>0)
            s = (np.array(st_img)>0)
            inter = (g & s).sum()
            union = (g | s).sum()
            iou = inter/union if union>0 else 0.0
            dice = (2*inter)/(g.sum()+s.sum()) if (g.sum()+s.sum())>0 else 0.0
            return iou, dice
        except Exception:
            g_data, size = bin_image(gt_img)
            s_data, _ = bin_image(st_img)
            inter = sum(1 for gv,sv in zip(g_data,s_data) if gv==1 and sv==1)
            g_sum = sum(g_data)
            s_sum = sum(s_data)
            union = g_sum + s_sum - inter
            iou = inter/union if union>0 else 0.0
            dice = (2*inter)/(g_sum+s_sum) if (g_sum+s_sum)>0 else 0.0
            return iou, dice
    if gtseg:
        results = []
        for gpath in gtseg:
            idx = parse_idx_from_gt(gpath)
            if idx is None or idx>=len(st):
                continue
            g = load_gray(gpath)
            s = load_gray(st[idx])
            iou, dice = metrics(g, s)
            results.append((idx, iou, dice))
        if results:
            mean_iou = sum(x[1] for x in results)/len(results)
            mean_dice = sum(x[2] for x in results)/len(results)
            lines = [f"idx={i:03d} IoU={iou:.3f} Dice={dice:.3f}" for i,iou,dice in results[:10]]
            print("accuracy_red_vs_green", f"mean_iou={mean_iou:.3f}", f"mean_dice={mean_dice:.3f}")
            for ln in lines:
                print("sample", ln)
            data = {
                "MeanIntersectionOverUnion": round(mean_iou, 3),
                "MeanDiceCoefficient": round(mean_dice, 3),
                "Samples": [
                    {"FrameIndex": i, "IntersectionOverUnion": round(iou, 3), "DiceCoefficient": round(dice, 3)}
                    for i,iou,dice in results
                ]
            }
            with open(os.path.join(outdir, "metrics.json"), "w") as fh:
                json.dump(data, fh, indent=2)
    print(outdir)
if __name__ == "__main__":
    main()

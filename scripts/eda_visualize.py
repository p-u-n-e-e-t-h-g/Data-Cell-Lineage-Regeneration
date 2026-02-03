import os
import sys
def main():
    sys.path.append("c:\\DATA CELL LINEAGE REGENERATION")
    from src.data.loader import PhCU373Dataset
    root = os.path.join("c:\\DATA CELL LINEAGE REGENERATION","Datasets","PhC-C2DH-U373")
    ds = PhCU373Dataset(root)
    frames = ds.list_frames()
    st = ds.list_st_masks()
    err = ds.list_err_masks()
    gtseg = ds.list_gt_seg()
    gttra = ds.list_gt_tra()
    tracks = ds.parse_tracks()
    print("frames", len(frames), frames[:3], frames[-3:] if frames else [])
    print("st_masks", len(st), st[:3], st[-3:] if st else [])
    print("err_masks", len(err), err[:3], err[-3:] if err else [])
    print("gt_seg", len(gtseg), gtseg[:3], gtseg[-3:] if gtseg else [])
    print("gt_tra", len(gttra), gttra[:3], gttra[-3:] if gttra else [])
    print("tracks_sample", tracks[:8])
    try:
        from PIL import Image
        im = Image.open(frames[0])
        ms = Image.open(st[0])
        print("image_size", im.size, "mask_size", ms.size)
    except Exception as e:
        print("image_lib_unavailable", type(e).__name__)
if __name__ == "__main__":
    main()

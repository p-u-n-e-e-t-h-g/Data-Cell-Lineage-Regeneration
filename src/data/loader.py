import os
from typing import List, Tuple, Optional
from PIL import Image
class PhCU373Dataset:
    def __init__(self, root: str):
        self.root = root
        self.seq = os.path.join(root, "01")
        self.st_seg = os.path.join(root, "01_ST", "SEG")
        self.err_seg = os.path.join(root, "01_ERR_SEG")
        self.gt_seg = os.path.join(root, "01_GT", "SEG")
        self.gt_tra = os.path.join(root, "01_GT", "TRA")
    def list_frames(self) -> List[str]:
        files = [f for f in os.listdir(self.seq) if f.lower().endswith(".tif")]
        files.sort()
        return [os.path.join(self.seq, f) for f in files]
    def list_st_masks(self) -> List[str]:
        files = [f for f in os.listdir(self.st_seg) if f.lower().endswith(".tif")]
        files.sort()
        return [os.path.join(self.st_seg, f) for f in files]
    def list_err_masks(self) -> List[str]:
        files = [f for f in os.listdir(self.err_seg) if f.lower().endswith(".tif")]
        files.sort()
        return [os.path.join(self.err_seg, f) for f in files]
    def list_gt_seg(self) -> List[str]:
        files = [f for f in os.listdir(self.gt_seg) if f.lower().endswith(".tif")]
        files.sort()
        return [os.path.join(self.gt_seg, f) for f in files]
    def list_gt_tra(self) -> List[str]:
        files = [f for f in os.listdir(self.gt_tra) if f.lower().endswith(".tif")]
        files.sort()
        return [os.path.join(self.gt_tra, f) for f in files]
    def parse_tracks(self) -> List[Tuple[int,int,int,int]]:
        txt = os.path.join(self.gt_tra, "man_track.txt")
        out = []
        with open(txt, "r") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) != 4:
                    continue
                cid = int(parts[0])
                start = int(parts[1])
                end = int(parts[2])
                parent = int(parts[3])
                out.append((cid, start, end, parent))
        return out
    def frame_index_from_name(self, name: str) -> Optional[int]:
        b = os.path.basename(name)
        if b.startswith("t") and b.endswith(".tif"):
            try:
                return int(b[1:-4])
            except:
                return None
        return None
    def get_frame(self, idx: int) -> Image.Image:
        return Image.open(self.list_frames()[idx]).convert("L")
    def get_st_mask(self, idx: int) -> Image.Image:
        return Image.open(self.list_st_masks()[idx]).convert("L")
    def get_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        return self.get_frame(idx), self.get_st_mask(idx)

# screen_yolo_min.py — Minimal screen capture + YOLO overlay
import argparse
import time
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import torch

# ---- Class category remap ----
CATEGORY_MAP = {
    3: "enemy", 4: "enemy", 5: "enemy",
    10: "enemy", 14: "enemy", 17: "enemy",
    18: "enemy", 22: "enemy", 25: "enemy",
    27: "enemy", 29: "enemy", 31: "enemy",
    33: "enemy", 36: "enemy", 37: "enemy",
    42: "enemy", 44: "enemy",

    1: "friendly", 8: "friendly", 16: "friendly",
    20: "friendly", 24: "friendly", 28: "friendly",
    35: "friendly", 39: "friendly",

    0: "friendly", 6: "friendly", 7: "friendly",
    11: "friendly", 13: "friendly", 15: "friendly",
    19: "friendly", 23: "friendly", 26: "friendly",
    30: "friendly", 32: "friendly", 34: "friendly",
    38: "friendly", 43: "friendly", 45: "friendly",

    2: "Spike", 9: "enemy_dead", 12: "friendly_dead",
    21: "last_seen", 40: "spike", 41: "spike_planted"
}

# ---- Simple IOU tracker (tracking-by-detection) ----
from dataclasses import dataclass
from typing import List, Tuple, Optional

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

_NEXT_TRACK_ID = 1

@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]  # xyxy
    cls_id: int
    label: str
    conf: float
    age: int = 0
    time_since_update: int = 0
    hit_streak: int = 0

    def update(self, new_bbox, conf, cls_id, label):
        self.bbox = new_bbox
        self.conf = conf
        self.cls_id = cls_id
        self.label = label
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1

class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[Track] = []
        self._next_id = 1

    def _new_track(self, bbox, conf, cls_id, label) -> Track:
        t = Track(track_id=self._next_id, bbox=bbox, cls_id=cls_id, label=label, conf=conf)
        self._next_id += 1
        return t

    def update(self, detections: List[Tuple[Tuple[float, float, float, float], float, int, str]]) -> List[Track]:
        """
        detections: list of (bbox_xyxy, conf, cls_id, label)
        Returns active tracks after update.
        """
        # 1) Mark all existing tracks as missed
        for tr in self.tracks:
            tr.mark_missed()

        if len(detections) == 0:
            # Prune old tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self.tracks

        # 2) Compute IOU between current tracks and detections
        iou_mat = [[0.0 for _ in range(len(detections))] for _ in range(len(self.tracks))]
        for ti, tr in enumerate(self.tracks):
            for di, (dbox, _, _, _) in enumerate(detections):
                iou_mat[ti][di] = iou_xyxy(tr.bbox, dbox)

        # 3) Greedy matching by IOU
        matched_tr = set()
        matched_det = set()
        # Build all (iou, ti, di) and sort descending
        pairs = []
        for ti in range(len(self.tracks)):
            for di in range(len(detections)):
                pairs.append((iou_mat[ti][di], ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for iou_val, ti, di in pairs:
            if iou_val < self.iou_threshold:
                break
            if ti in matched_tr or di in matched_det:
                continue
            # Match
            dbox, dconf, dcls, dlabel = detections[di]
            self.tracks[ti].update(dbox, dconf, dcls, dlabel)
            matched_tr.add(ti)
            matched_det.add(di)

        # 4) Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            dbox, dconf, dcls, dlabel = det
            self.tracks.append(self._new_track(dbox, dconf, dcls, dlabel))

        # 5) Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks
# ---- End tracker ----

WIN_NAME = "YOLO Live"

def pick_device():
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="bestWeight.pt", help="Path to trained YOLO checkpoint")
    ap.add_argument("--monitor", type=int, default=2, help="1=primary, 2=second, ...")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou",  type=float, default=0.45)
    ap.add_argument("--downscale", type=float, default=1.0, help="e.g., 0.75 for speed")
    ap.add_argument("--isolation_px", type=float, default=40.0, help="Mark friendly as isolated if nearest friendly is farther than this many pixels")
    args = ap.parse_args()

    device = pick_device()
    model = YOLO(args.weights).to(device)
    tracker = IOUTracker(iou_threshold=0.3, max_age=20)

    sct = mss()
    mons = sct.monitors
    # Clamp monitor index
    mon_idx = max(1, min(args.monitor, len(mons) - 1))
    monitor = mons[mon_idx]
    print("Detected monitors:")
    for i, m in enumerate(mons):
        tag = "VIRTUAL" if i == 0 else f"MONITOR {i}"
        print(f"{i}: {tag} -> {m}")
    print(f"Using MONITOR {mon_idx}: {monitor}")
    print("Press ESC to quit.")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WIN_NAME, 60, 40)

    last = time.time()
    f = 0

    while True:
        # Grab screen (BGRA) and convert to BGR
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)[:, :, :3]           # drop alpha
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # Crop to minimap region (hardcoded ROI)
        # Minimap crop region: x=23, y=0, w=632, h=598
        frame = frame[110:110+265, 0:0+265]

        # Optional downscale for speed
        if args.downscale != 1.0:
            frame = cv2.resize(frame, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_LINEAR)

        # YOLO inference
        results = model(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            agnostic_nms=False,
            verbose=False,
            device=device,
        )

        # === Build detections from YOLO outputs ===
        boxes = results[0].boxes
        raw_names = getattr(results[0], "names", None) or getattr(model, "names", {})
        names = dict(enumerate(raw_names)) if isinstance(raw_names, (list, tuple)) else (raw_names or {})

        det_list: List[Tuple[Tuple[float, float, float, float], float, int, str]] = []
        if boxes is not None and getattr(boxes, "xyxy", None) is not None and boxes.xyxy.numel() > 0:
            xyxy  = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
            clses = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else None
            n = xyxy.shape[0]
            for i in range(n):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i]) if confs is not None else float("nan")
                cls_id = int(clses[i]) if clses is not None else -1
                category = CATEGORY_MAP.get(cls_id, "other")
                label = names.get(cls_id, str(cls_id))
                det_list.append(((float(x1), float(y1), float(x2), float(y2)), conf, cls_id, label))

        # === Update tracker with current detections ===
        tracks = tracker.update(det_list)

        # === Isolation check for friendly tracks ===
        # Mark a friendly track as isolated if its nearest other friendly is > args.isolation_px away
        friendly_tracks = [tr for tr in tracks if CATEGORY_MAP.get(tr.cls_id, "other") == "friendly"]
        isolated_ids = set()
        if len(friendly_tracks) >= 1:
            centers = {}
            for tr in friendly_tracks:
                x1, y1, x2, y2 = tr.bbox
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                centers[tr.track_id] = (cx, cy)
            for tr in friendly_tracks:
                cx, cy = centers[tr.track_id]
                # compute min distance to any other friendly
                min_d = None
                for other in friendly_tracks:
                    if other.track_id == tr.track_id:
                        continue
                    ox, oy = centers[other.track_id]
                    dx = cx - ox
                    dy = cy - oy
                    d = (dx * dx + dy * dy) ** 0.5
                    if (min_d is None) or (d < min_d):
                        min_d = d
                # If there is no other friendly, or the nearest is too far -> isolated
                if (min_d is None) or (min_d > args.isolation_px):
                    isolated_ids.add(tr.track_id)

        # === Draw (BGR) with stable track IDs ===
        annotated = frame.copy()
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            display_title = CATEGORY_MAP.get(tr.cls_id, "other")
            # Base color by category
            if display_title == "friendly":
                color = (0, 255, 0)  # green
            elif display_title == "enemy":
                color = (0, 0, 255)  # red
            else:
                color = (255, 0, 0)  # blue

            # Override color for isolated friendly
            isolated = (display_title == "friendly") and (tr.track_id in isolated_ids)
            if isolated:
                color = (0, 255, 255)  # yellow
            
            cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), color, 2)
            suffix = " (isolated)" if isolated else ""
            # label_text = f"#{tr.track_id} {display_title}{suffix} {tr.conf:.2f}"
            label_text = ""
            cv2.putText(annotated, label_text, (x1i, max(y1i-6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Optional: print current active tracks (one line per track)
        # for tr in tracks:
        #     disp = CATEGORY_MAP.get(tr.cls_id, "other")
        #     print(f"track {tr.track_id}: {disp} ({tr.label}) conf={tr.conf:.2f} bbox=[{tr.bbox[0]:.1f},{tr.bbox[1]:.1f},{tr.bbox[2]:.1f},{tr.bbox[3]:.1f}]")

        # FPS overlay
        # f += 1
        # if f % 10 == 0:
        #     now = time.time()
        #     fps = 10.0 / (now - last)
        #     last = now
        #     cv2.putText(annotated, f"{fps:.1f} FPS ({device})", (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
"""Offline RGB/depth visualization; no thermal or physiological inference."""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from calibration import load_paras_txt, get_camera, get_intrinsics
from depth_utils import gray_to_depth, depth_in_roi
from detection import CompositeDetector


def run(dataset_dir, camera_index=0, max_frames=0, no_display=True, no_person=False,
        units_to_meters=None, output=None, suffix="mp4"):
    if max_frames < 0:
        raise ValueError("max_frames must be nonnegative")
    root = Path(dataset_dir)
    camera = get_camera(load_paras_txt(root / "Camera Parameters/paras.txt"), camera_index)
    K, width, height = get_intrinsics(camera)
    prefix = f"camera{camera_index+1:02d}"
    paths = [root / f"{prefix}-{kind}.{suffix}" for kind in ("rgb", "depth")]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f"missing RGB/depth pair for {prefix}")
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    writer = None
    start, count, detections_count = time.perf_counter(), 0, 0
    try:
        if not all(c.isOpened() for c in caps):
            raise ValueError("video decoder could not open pair")
        fps = [c.get(cv2.CAP_PROP_FPS) for c in caps]
        if not all(math.isfinite(f) and f > 0 for f in fps) or not math.isclose(*fps, rel_tol=1e-4):
            raise ValueError("paired videos need known matching FPS")
        sizes = [(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for c in caps]
        if any(s != (width, height) for s in sizes):
            raise ValueError("RGB/depth dimensions do not match calibration")
        detector = CompositeDetector(detect_person=not no_person)
        if output:
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps[0], (2*width, height))
            if not writer.isOpened():
                raise ValueError("output video writer failed")
        while max_frames == 0 or count < max_frames:
            ok_rgb, rgb = caps[0].read(); ok_dep, dep = caps[1].read()
            if ok_rgb != ok_dep:
                raise ValueError("one paired stream ended early")
            if not ok_rgb:
                break
            if rgb.shape[:2] != (height,width) or dep.shape[:2] != (height,width):
                raise ValueError("decoded resolution changed")
            dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY) if dep.ndim == 3 else dep
            depth = gray_to_depth(dep, encoding="byte", units_to_meters=units_to_meters)
            detections = detector.detect(rgb); detections_count += len(detections)
            vis = rgb.copy()
            for box, label, score in detections:
                d = depth_in_roi(depth, box)
                unit = "m" if units_to_meters is not None else "dataset units"
                annotation = f"{label} {d:.2f} {unit}" if d is not None else f"{label} depth unavailable"
                cv2.rectangle(vis, box[:2], box[2:], (0,255,0), 1)
                cv2.putText(vis, annotation, (box[0],max(12,box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,255,0), 1)
            depvis = cv2.applyColorMap(dep, cv2.COLORMAP_INFERNO)
            combined = np.hstack((vis, depvis))
            if writer is not None:
                writer.write(combined)
            count += 1
            if not no_display:
                cv2.imshow("RGB and encoded depth", combined)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()
    return {"camera_video_prefix":prefix,"calibration_camera_id":camera["camera_id"],
            "frames":count,"detections":detections_count,"source_fps":fps[0],
            "depth_units":"metres" if units_to_meters is not None else "dataset units",
            "metric_scale":units_to_meters,"wall_seconds":time.perf_counter()-start,
            "alignment":"assumed paired frame index; matching FPS is not sensor synchronization proof"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--camera-index",type=int,default=0)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--no-display",action="store_true")
    p.add_argument("--no-person",action="store_true")
    p.add_argument("--units-to-meters",type=float)
    p.add_argument("--out",type=Path)
    p.add_argument("--suffix",choices=("mp4","avi"),default="mp4")
    p.add_argument("--report",type=Path)
    args=p.parse_args()
    report=run(args.dataset_dir,args.camera_index,args.max_frames,args.no_display,args.no_person,args.units_to_meters,args.out,args.suffix)
    text=json.dumps(report,indent=2,allow_nan=False)
    print(text)
    if args.report:args.report.write_text(text+"\n")


if __name__=="__main__":main()

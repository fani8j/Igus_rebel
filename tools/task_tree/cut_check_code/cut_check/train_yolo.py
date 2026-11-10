"""Train the YOLOv8 segmentation model for cut / partial / not-cut / knife.

Needs the dataset from make_yolo_dataset.py. Starts from the small pretrained
YOLOv8n-seg model and fine-tunes it. On a laptop CPU this takes about an hour;
with an NVIDIA GPU use --device 0.

Usage:
    python train_yolo.py                  # 30 epochs on the CPU
    python train_yolo.py --epochs 60 --device 0

The best weights are copied to models/cut_seg_yolov8n.pt.
"""
import argparse
import os
import shutil

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu, or a GPU number such as 0")
    ap.add_argument("--weights", default="yolov8n-seg.pt",
                    help="start from these weights, e.g. models/cut_seg_yolov8n.pt to fine-tune")
    args = ap.parse_args()

    project = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project)
    data = os.path.join(project, "dataset_yolo", "data.yaml")
    if not os.path.isfile(data):
        raise SystemExit("dataset_yolo/data.yaml not found - run make_yolo_dataset.py first")

    model = YOLO(args.weights)  # yolov8n-seg.pt is downloaded automatically the first time
    # amp=False: half precision is very slow on older GPUs such as the GeForce MX250
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=2, cache=True, amp=False, project=os.path.join(project, "runs"), name="cut_seg",
                exist_ok=True, plots=True, verbose=False)

    best = os.path.join(project, "runs", "cut_seg", "weights", "best.pt")
    os.makedirs(os.path.join(project, "models"), exist_ok=True)
    target = os.path.join(project, "models", "cut_seg_yolov8n.pt")
    shutil.copy(best, target)
    metrics = YOLO(target).val(data=data, imgsz=args.imgsz, device=args.device, plots=False, verbose=False)
    names = metrics.names
    print("\nvalidation (images never used for training), mask mAP50 per class:")
    for i, c in enumerate(metrics.seg.ap_class_index):
        print(f"  {names[int(c)]:12s} {metrics.seg.ap50[i]:.3f}")
    print(f"  all classes  {metrics.seg.map50:.3f}")
    print(f"saved {target}")


if __name__ == "__main__":
    main()

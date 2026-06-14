import argparse


def main():
    parser = argparse.ArgumentParser(description="Option A re-evaluation with TTA + hi-res")
    parser.add_argument("--model", default="backend/model/weights/best.pt")
    parser.add_argument("--data", default="backend/model/rdd2022.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--name", default="val_optionA")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"Validating {args.model} on '{args.split}' @ imgsz={args.imgsz}, "
          f"TTA={args.tta}, iou={args.iou} ...")

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        augment=args.tta,     # test-time augmentation
        iou=args.iou,
        split=args.split,
        plots=True,           # regenerates PR / F1 / P / R curves + confusion matrix
        name=args.name,
    )

    print("\n=== Option A results (paste into report Table 4) ===")
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    print("\n=== Per-class AP@50 (paste into Table 5) ===")
    names = model.names
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {names[i]:20s}: {ap:.4f}")

    print(f"\nCurves + confusion matrix saved under runs/detect/{args.name}/")


if __name__ == "__main__":
    main()

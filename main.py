import argparse

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = "chess_pieces.pt"


def main():
    parser = argparse.ArgumentParser(description="Detect chess pieces with YOLO")
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Camera index (0), video path, or stream URL",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chess piece model weights file")
    parser.add_argument(
        "--conf", type=float, default=0.6, help="Confidence threshold (default 0.6)"
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Chess Piece Detection", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

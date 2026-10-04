#!/usr/bin/env python3
"""Teste le modèle YOLO en temps réel sur les caméras (OAK-D Lite "top" et caméra poignet USB "wrist").

Tourne sur l'hôte dans yolo/.venv (depthai + ultralytics), avec les mêmes résolutions que capture_globe.py
(top 1920x1080, wrist 1280x720). Avec le bras : make yolo-teleop lance aussi la téléopération leader -> follower.

Fenêtre : Q/Échap = quitter, S = snapshot (brut + annoté), ESPACE = pause, +/- = seuil de confiance.

Usage :
    python yolo/live.py                                    # les deux caméras, best.pt du run "globe"
    python yolo/live.py --cameras oak --conf 0.4
    python yolo/live.py --target "48°28"                   # met en évidence un repère (sous-chaîne du nom)
    python yolo/live.py --weights yolo/runs/detect/globe2/weights/best.pt --wrist-device /dev/video4
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts/camera"))
from camera_utils import OAKCamera, V4L2Camera  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=HERE / "runs/detect/globe/weights/best.pt")
    p.add_argument("--cameras", choices=["both", "oak", "wrist"], default="both")
    p.add_argument("--wrist-device", default=os.environ.get("WRIST_DEVICE") or "/dev/video2")
    p.add_argument("--top-size", default="1920x1080")
    p.add_argument("--wrist-size", default="1280x720")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640, help="Même valeur qu'à l'entraînement")
    p.add_argument("--device", default=None, help="0, cpu... (défaut : GPU si dispo)")
    p.add_argument("--target", help="Sous-chaîne du nom de classe à mettre en évidence (ex. 48°28)")
    p.add_argument("--save", type=Path, default=HERE / "runs/live")
    return p.parse_args()


class LatestFrame(threading.Thread):
    """Lit une caméra en continu dans un thread et ne garde que le dernier frame : une caméra lente ou
    l'inférence ne retarde jamais l'autre caméra."""

    def __init__(self, name: str, cam):
        super().__init__(daemon=True, name=name)
        self.cam, self.frame, self.stamp, self.running = cam, None, 0.0, True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ok, frame = self.cam.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame, self.stamp = frame, time.time()
            else:
                time.sleep(0.01)

    def latest(self):
        with self.lock:
            return self.frame, self.stamp

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.cam.release()


def open_cameras(args) -> dict[str, LatestFrame]:
    readers = {}
    if args.cameras in ("both", "oak"):
        w, h = map(int, args.top_size.split("x"))
        print(f"OAK-D Lite {w}x{h}...")
        readers["top"] = LatestFrame("top", OAKCamera(width=w, height=h).open())
    if args.cameras in ("both", "wrist"):
        w, h = map(int, args.wrist_size.split("x"))
        print(f"Caméra poignet {args.wrist_device} {w}x{h}...")
        cam = V4L2Camera(device=args.wrist_device, width=w, height=h).open()
        if not cam.isOpened():
            sys.exit(f"Caméra poignet inaccessible : {args.wrist_device} (v4l2-ctl --list-devices, puis --wrist-device)")
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        readers["wrist"] = LatestFrame("wrist", cam)
    for r in readers.values():
        r.start()
    return readers


def draw(frame: np.ndarray, result, names: dict, target_ids: set[int], name: str) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    lw = max(2, round(w / 640))
    fs = 0.5 * w / 1280 + 0.2
    boxes = result.boxes
    for (x1, y1, x2, y2), c, p in zip(boxes.xyxy.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist()):
        is_target = c in target_ids
        color = (0, 0, 255) if is_target else colors(c, bgr=True)
        thick = lw * 2 if is_target else lw
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 4 * lw, lw)
        label = f"{names[c]} {p:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = y1 - 6 if y1 - th - 8 > 0 else y2 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), color, -1)
        text_color = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)
        cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, text_color, 1, cv2.LINE_AA)
        if is_target:  # écart au centre de l'image : ce qu'il faudra annuler pour viser le repère
            cv2.line(img, (w // 2, h // 2), (cx, cy), (0, 0, 255), lw)
            cv2.putText(img, f"dx={cx - w // 2:+d} dy={cy - h // 2:+d}px", (cx + 8, cy + 8 + th * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 2, cv2.LINE_AA)
    if target_ids:
        cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 8 * lw, lw)
    cv2.putText(img, f"{name}  {len(boxes)} detection(s)", (10, int(30 * fs / 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                fs * 1.4, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def tile(images: list[np.ndarray], height: int = 540) -> np.ndarray:
    resized = [cv2.resize(im, (round(im.shape[1] * height / im.shape[0]), height)) for im in images]
    return np.hstack(resized)


def window_closed(win: str) -> bool:
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:  # backend Qt : la fenêtre peut ne pas être encore prête au premier tour
        return False


def main():
    args = parse_args()
    if not args.weights.is_file():
        sys.exit(f"Poids introuvables : {args.weights} (make yolo-train, ou --weights)")
    model = YOLO(args.weights)
    names = model.names
    target_ids = {i for i, n in names.items() if args.target and args.target in n}
    if args.target and not target_ids:
        sys.exit(f"Aucune classe ne contient '{args.target}'. Classes : {', '.join(names.values())}")
    model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=args.imgsz, device=args.device, verbose=False)  # warmup

    readers = open_cameras(args)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # le Qt embarqué d'opencv-python n'a que le plugin X11
    win = "YOLO live - Q: quitter  S: snapshot  ESPACE: pause  +/-: confiance"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    conf, paused, fps, last = args.conf, False, 0.0, time.time()
    shown, frames = None, {}
    try:
        while True:
            if not paused:
                frames = {n: r.latest()[0] for n, r in readers.items()}
                frames = {n: f for n, f in frames.items() if f is not None}
                if frames:
                    t0 = time.time()
                    results = model.predict(list(frames.values()), imgsz=args.imgsz, conf=conf,
                                            device=args.device, verbose=False)
                    infer_ms = (time.time() - t0) * 1000
                    panels = [draw(f, r, names, target_ids, n) for (n, f), r in zip(frames.items(), results)]
                    shown = tile(panels)
                    now = time.time()
                    fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                    last = now
                    cv2.putText(shown, f"{fps:4.1f} FPS  inference {infer_ms:4.0f} ms  conf>={conf:.2f}",
                                (10, shown.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if shown is None:
                shown = np.zeros((540, 960, 3), np.uint8)
                cv2.putText(shown, "Attente des cameras...", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(win, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or window_closed(win):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("+"), ord("=")):
                conf = min(conf + 0.05, 0.95)
            elif key == ord("-"):
                conf = max(conf - 0.05, 0.05)
            elif key == ord("s") and frames:
                args.save.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                for n, f in frames.items():
                    cv2.imwrite(str(args.save / f"{n}_{stamp}.jpg"), f)  # brut : réutilisable pour annoter
                cv2.imwrite(str(args.save / f"annotated_{stamp}.jpg"), shown)
                print(f"Snapshot -> {args.save}/*_{stamp}.jpg")
    finally:
        for r in readers.values():
            r.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

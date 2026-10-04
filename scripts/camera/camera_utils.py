"""Utilitaires caméra partagés : correction de balance des blancs (LAB) et wrappers caméra compatibles cv2.VideoCapture (flux ZMQ oak_zmq_server, accès direct depthai, webcam USB générique V4L2)"""

import json
import os
import sys
import time

import cv2
import numpy as np

_CAMERA_CONFIG = "datasets/camera_config.json"


def white_balance(frame):
    """Applique la correction de balance des blancs sauvegardée à un frame BGR. no-op si non calibré (offsets à 0)"""
    wb = {"a_offset": 0.0, "b_offset": 0.0}
    if os.path.exists(_CAMERA_CONFIG):
        with open(_CAMERA_CONFIG) as f:
            wb.update(json.load(f).get("white_balance", {}))
    # LAB plutôt que BGR : décorrèle luminance (L) et couleur (a/b), donc décaler a/b corrige la teinte sans toucher à la luminosité de l'image.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] += wb["a_offset"]
    lab[:, :, 2] += wb["b_offset"]
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


class ZMQCamera:
    """Souscrit au flux ZMQ de oak_zmq_server.py, même interface que OAKCamera."""

    def __init__(self, port=5555, camera_name="top", timeout_ms=2000):
        self.port = port
        self.camera_name = camera_name
        self.timeout_ms = timeout_ms
        self._socket = None
        self._context = None

    def open(self):
        import base64
        import zmq
        self._base64 = base64
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        # CONFLATE=1 : ne garde que le dernier message reçu, jamais de backlog.
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.connect(f"tcp://localhost:{self.port}")
        return self

    def isOpened(self):
        return self._socket is not None

    def read(self):
        if self._socket is None:
            return False, None
        try:
            msg = json.loads(self._socket.recv_string())
            buf = self._base64.b64decode(msg["images"][self.camera_name])
            arr = np.frombuffer(buf, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return False, None
            # oak_zmq_server.py envoie du RGB : reconvertir en BGR ici
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            return False, None

    def set(self, *args):
        pass

    def release(self):
        if self._socket is not None:
            self._socket.close()
            self._context.term()
            self._socket = None
            self._context = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()


class V4L2Camera:
    """Webcam USB générique (ET-S231 et similaires) via OpenCV. drop-in pour cv2.VideoCapture.
    V4L2 (/dev/videoN) sous Linux ; backend par défaut sous macOS/Windows, où /dev/videoN
    n'existe pas (N réutilisé comme index de caméra, à ajuster si ce n'est pas la bonne)."""

    def __init__(self, device="/dev/video0", width=1920, height=1080, fps=30, fourcc="MJPG"):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self._cap = None

    def open(self):
        device, backend = self.device, cv2.CAP_V4L2
        if not sys.platform.startswith("linux"):
            backend = cv2.CAP_ANY
            if isinstance(device, str) and device.startswith("/dev/video"):
                device = int(device.removeprefix("/dev/video"))
        self._cap = cv2.VideoCapture(device, backend)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self

    def isOpened(self):
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        if self._cap is None:
            return False, None
        return self._cap.read()

    def set(self, *args):
        if self._cap is not None:
            self._cap.set(*args)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()


class OAKCamera:
    """OAK-D Lite via depthai v3. drop-in pour cv2.VideoCapture"""

    def __init__(self, width=1920, height=1080, fps=30, warmup=30):
        import depthai as dai
        self._dai = dai
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self._pipeline = None
        self._queue = None

    def open(self):
        dai = self._dai
        self._pipeline = dai.Pipeline()
        pipeline = self._pipeline.__enter__()
        cam = pipeline.create(dai.node.Camera).build()
        self._queue = cam.requestOutput(
            (self.width, self.height), fps=self.fps
        ).createOutputQueue()
        pipeline.start()
        # Frames jetées le temps que l'auto-exposition/focus de l'OAK-D se stabilise
        for _ in range(self.warmup):
            self._queue.get()
        return self

    def isOpened(self):
        return self._queue is not None

    def read(self):
        if self._queue is None:
            return False, None
        try:
            return True, self._queue.get().getCvFrame()
        except Exception:
            return False, None

    def read_latest(self):
        """Comme read(), mais vide la file et renvoie le frame le plus récent (évite un frame en retard
        de plusieurs centaines de ms si la boucle appelante est plus lente que la caméra)"""
        if self._queue is None:
            return False, None
        try:
            msgs = self._queue.tryGetAll()
            msg = msgs[-1] if msgs else self._queue.get()
            return True, msg.getCvFrame()
        except Exception:
            return False, None

    def set(self, *args):
        pass

    def release(self):
        if self._pipeline is not None:
            try:
                if self._queue is not None:
                    self._queue.close()
                self._pipeline.stop()
                time.sleep(0.5)
                self._pipeline.__exit__(None, None, None)
            except Exception:
                pass
            self._pipeline = None
            self._queue = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()

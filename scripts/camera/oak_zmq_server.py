"""OAK-D Lite -> ZMQ bridge pour lerobot-record (type: zmq, camera_name: top)"""
import base64
import json
import os
import sys
import time

import cv2
import depthai as dai
import zmq

CAMERA_NAME = "top"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
# 640x360 par défaut : en 1920x1080, JPEG + base64 + décodage + encodage AV1 en direct saturent le conteneur,
# la boucle lerobot-record tombe à ~3 Hz et la lecture des servos expire ("There is no status packet!").
WIDTH = int(os.environ.get("OAK_WIDTH", 640))
HEIGHT = int(os.environ.get("OAK_HEIGHT", 360))
FPS = 30

context = zmq.Context()
socket = context.socket(zmq.PUB)
# SNDHWM=20 : borne la queue d'envoi
socket.setsockopt(zmq.SNDHWM, 20)
# LINGER=0 : close() ne bloque pas à attendre l'envoi des messages en attente
socket.setsockopt(zmq.LINGER, 0)
socket.bind(f"tcp://*:{PORT}")
print(f"OAK ZMQ server port={PORT} camera_name={CAMERA_NAME} {WIDTH}x{HEIGHT}")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    queue = cam.requestOutput((WIDTH, HEIGHT), fps=FPS).createOutputQueue()
    pipeline.start()
    for _ in range(10):
        queue.get()
    print("OAK-D Lite prêt, streaming...")
    try:
        while pipeline.isRunning():
            t0 = time.time()
            bgr = queue.get().getCvFrame()
            # lerobot ZMQCamera (color_mode=RGB) decode sans reconvertir : swap ici
            # pour que le frame reçu côté lerobot-record soit dans le bon ordre R/G/B.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode()
            msg = {"timestamps": {CAMERA_NAME: time.time()}, "images": {CAMERA_NAME: b64}}
            try:
                # NOBLOCK + Again ignoré : si aucun abonné n'est prêt à recevoir, abandon de la frame
                socket.send_string(json.dumps(msg), zmq.NOBLOCK)
            except zmq.Again:
                pass
            sleep = (1.0 / FPS) - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()

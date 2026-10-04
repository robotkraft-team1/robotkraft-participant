#!/usr/bin/env python3

import argparse
import time
import cv2
import depthai as dai
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="DepthAI OAK-D Camera & Real-Time Distance Viewer"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "rgb", "depth"],
        default="both",
        help="Display mode: 'both' (RGB + Colormapped Depth), 'rgb', or 'depth'. Default: 'both'",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target frame rate (FPS). Default: 30.0",
    )
    parser.add_argument(
        "--extended-disparity",
        action="store_true",
        help="Closer minimum distance threshold (good for objects < 35 cm)",
    )
    parser.add_argument(
        "--subpixel",
        action="store_true",
        default=True,
        help="Enable subpixel disparity for better precision (Default: True)",
    )
    return parser.parse_args()


# Global for mouse coordinate tracking
mouse_x, mouse_y = -1, -1


def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def main():
    global mouse_x, mouse_y
    args = parse_args()

    # Small pause to allow device to settle
    time.sleep(1)

    pipeline = dai.Pipeline()

    # Color Camera (CAM_A - IMX214)
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 2)  # Output size: 960x540
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setFps(args.fps)

    # Left Mono Camera (CAM_B - OV7251)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setFps(args.fps)

    # Right Mono Camera (CAM_C - OV7251)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setFps(args.fps)

    # Stereo Depth Engine
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(args.subpixel)
    stereo.setExtendedDisparity(args.extended_disparity)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth map directly with RGB camera

    # Link nodes
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Output streams
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    xout_disparity = pipeline.create(dai.node.XLinkOut)
    xout_disparity.setStreamName("disparity")
    stereo.disparity.link(xout_disparity.input)

    print("[INFO] Connecting to device...")
    with dai.Device(pipeline) as device:
        usb_speed = device.getUsbSpeed()
        print(f"[INFO] Connected to {device.getMxId()} (USB Speed: {usb_speed.name})")

        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        q_disp = device.getOutputQueue("disparity", maxSize=4, blocking=False)

        rgb_win_name = "OAK-D - RGB & Distance"
        depth_win_name = "OAK-D - Depth Map (Heatmap)"

        if args.mode in ["both", "rgb"]:
            cv2.namedWindow(rgb_win_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(rgb_win_name, on_mouse)

        if args.mode in ["both", "depth"]:
            cv2.namedWindow(depth_win_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(depth_win_name, on_mouse)

        fps = 0.0
        frame_count = 0
        start_time = time.time()

        print("\n" + "=" * 55)
        print("  STREAMING EN DIRECT AVEC MESURE DE DISTANCE")
        print("  - Visez avec le réticule central pour la distance")
        print("  - Survolez l'image avec la souris pour mesurer un point")
        print("  - Touche 'q' pour quitter")
        print("=" * 55 + "\n")

        try:
            while True:
                in_rgb = q_rgb.tryGet()
                in_depth = q_depth.tryGet()
                in_disp = q_disp.tryGet()

                if in_rgb is not None and in_depth is not None:
                    rgb_frame = in_rgb.getCvFrame()
                    depth_frame = in_depth.getFrame()  # uint16 (in mm)

                    h, w = depth_frame.shape
                    cx, cy = w // 2, h // 2

                    # Calculate real FPS
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        frame_count = 0
                        start_time = time.time()

                    # Distance measurement at center
                    center_dist_mm = depth_frame[cy, cx]
                    if 100 < center_dist_mm < 15000:
                        center_dist_str = f"{center_dist_mm / 1000.0:.2f} m ({center_dist_mm} mm)"
                    else:
                        center_dist_str = "Hors portee"

                    # Distance measurement at mouse cursor if inside frame
                    mouse_dist_str = None
                    if 0 <= mouse_x < w and 0 <= mouse_y < h:
                        m_dist_mm = depth_frame[mouse_y, mouse_x]
                        if 100 < m_dist_mm < 15000:
                            mouse_dist_str = f"{m_dist_mm / 1000.0:.2f} m"
                        else:
                            mouse_dist_str = "N/A"

                    # Draw crosshair at center
                    color_green = (0, 255, 0)
                    cv2.drawMarker(
                        rgb_frame,
                        (cx, cy),
                        color_green,
                        markerType=cv2.MARKER_CROSS,
                        markerSize=25,
                        thickness=2,
                    )

                    # Overlay info on RGB frame
                    cv2.putText(
                        rgb_frame,
                        f"Distance centre : {center_dist_str}",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        rgb_frame,
                        f"FPS: {fps:.1f} | USB: {usb_speed.name} | Res: {w}x{h}",
                        (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        1,
                    )

                    if mouse_dist_str:
                        cv2.circle(rgb_frame, (mouse_x, mouse_y), 5, (0, 0, 255), -1)
                        cv2.putText(
                            rgb_frame,
                            f"{mouse_dist_str}",
                            (mouse_x + 10, mouse_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                        )

                    # Colorize depth / disparity
                    if args.mode in ["both", "depth"]:
                        if in_disp is not None:
                            disp_frame = in_disp.getFrame()
                            # Colorize with JET colormap
                            max_disp = stereo.initialConfig.getMaxDisparity()
                            disp_visual = (disp_frame * (255.0 / max_disp)).astype(np.uint8)
                            depth_color = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)
                        else:
                            # Fallback normalize depth
                            depth_norm = cv2.normalize(
                                depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
                            )
                            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                        # Draw marker on depth frame as well
                        cv2.drawMarker(
                            depth_color,
                            (cx, cy),
                            (255, 255, 255),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=25,
                            thickness=2,
                        )
                        cv2.putText(
                            depth_color,
                            f"Distance: {center_dist_str}",
                            (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (255, 255, 255),
                            2,
                        )

                    # Display windows
                    if args.mode in ["both", "rgb"]:
                        cv2.imshow(rgb_win_name, rgb_frame)
                    if args.mode in ["both", "depth"]:
                        cv2.imshow(depth_win_name, depth_color)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    print("[INFO] Stopped cleanly.")


if __name__ == "__main__":
    main()

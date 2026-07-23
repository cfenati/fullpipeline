# blackfly_gain_control.py
import time
from datetime import datetime

import cv2
import numpy as np
import PySpin

GAIN_STEP = 1.0  # adjust to taste (e.g., 0.5)
SAVE_DIR = "/home/cfenati/projects/MasterThesis/FullPipeline/OutputFLIR"


def get_float_node(nodemap, name):
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    return node if (PySpin.IsAvailable(node) and PySpin.IsReadable(node)) else None


def set_color_processing(processor):
    for name in [
        "ColorProcessingAlgorithm_HQ_LINEAR",
        "HQ_LINEAR",
        "ColorProcessingAlgorithm_BILINEAR",
        "BILINEAR",
        "ColorProcessingAlgorithm_DEFAULT",
        "DEFAULT",
    ]:
        algo = getattr(PySpin, name, None)
        if algo is not None:
            try:
                processor.SetColorProcessing(algo)
                return
            except Exception:
                pass


def main():
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    if cams.GetSize() == 0:
        print("No cameras found.")
        cams.Clear()
        system.ReleaseInstance()
        return

    cam = cams[0]
    print("Initializing camera...")
    cam.Init()
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

    # Optional: Mono8 for simplicity/speed
    try:
        cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
    except Exception:
        pass

    # Manual gain
    try:
        if cam.GainAuto.GetAccessMode() == PySpin.RW:
            cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    except Exception:
        pass

    nodemap = cam.GetNodeMap()
    gain_node = get_float_node(nodemap, "Gain")
    if gain_node and PySpin.IsWritable(gain_node):
        gmin, gmax = gain_node.GetMin(), gain_node.GetMax()
        print(f"Gain range: {gmin:.2f} .. {gmax:.2f} (current {gain_node.GetValue():.2f})")
    else:
        print("Gain node not available/writable; '[' and ']' will be ignored.")
        gain_node = gmin = gmax = None

    processor = PySpin.ImageProcessor()
    set_color_processing(processor)

    print("Starting acquisition…  (q=quit, ]=gain+, [=gain-, s=save frame)")
    cam.BeginAcquisition()
    last_t = time.time()

    try:
        while True:
            img_ptr = cam.GetNextImage(1000)
            if img_ptr.IsIncomplete():
                img_ptr.Release()
                continue

            try:
                # Try BGR8 conversion
                conv = processor.Convert(img_ptr, PySpin.PixelFormat_BGR8)
                h, w = conv.GetHeight(), conv.GetWidth()
                frame = np.frombuffer(conv.GetData(), dtype=np.uint8).reshape(h, w, 3).copy()
            except Exception:
                # Fallback for raw Mono8
                h, w = img_ptr.GetHeight(), img_ptr.GetWidth()
                buf = np.frombuffer(img_ptr.GetData(), dtype=np.uint8)
                if buf.size == h * w:
                    frame = cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR).copy()
                else:
                    img_ptr.Release()
                    continue
            finally:
                img_ptr.Release()

            # FPS overlay
            now = time.time()
            dt = now - last_t
            last_t = now
            fps_text = f"{1.0/dt:.1f} FPS" if dt > 0 else "FPS: --"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if gain_node:
                cv2.putText(
                    frame,
                    f"Gain: {gain_node.GetValue():.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("FLIR Blackfly", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if gain_node:
                if key == ord("]"):
                    new_gain = min(gain_node.GetValue() + GAIN_STEP, gmax)
                    gain_node.SetValue(new_gain)
                    print(f"Gain increased to {new_gain:.2f}")
                elif key == ord("["):
                    new_gain = max(gain_node.GetValue() - GAIN_STEP, gmin)
                    gain_node.SetValue(new_gain)
                    print(f"Gain decreased to {new_gain:.2f}")
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = SAVE_DIR + f"/blackfly_{ts}.png"
                cv2.imwrite(str(fname), frame)
                print(f"Saved {fname}")

    finally:
        try:
            cam.EndAcquisition()
        except Exception:
            pass
        cam.DeInit()
        del cam
        cams.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()
        print("Closed camera.")


if __name__ == "__main__":
    main()

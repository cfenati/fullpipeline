import cv2
import time
from pathlib import Path

SAVE_DIR = Path("/home/cfenati/projects/MasterThesis/RGBcameras")

CAMERA_1 = "/dev/video6"
CAMERA_2 = "/dev/video8"

#WIDTH = 2320
#HEIGHT = 1744
#FPS = 25

WIDTH = 1280
HEIGHT = 720
FPS = 30


def open_camera(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {device}")

    # Set format before resolution and frame rate.
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(
        chr((fourcc_value >> (8 * i)) & 0xFF)
        for i in range(4)
    )

    print(
        f"{device}: "
        f"{actual_width}x{actual_height}, "
        f"{actual_fps:.2f} FPS, "
        f"{actual_fourcc}"
    )

    return cap


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cap1 = None
    cap2 = None

    try:
        cap1 = open_camera(CAMERA_1)
        cap2 = open_camera(CAMERA_2)

        while True:
            grabbed1 = cap1.grab()
            grabbed2 = cap2.grab()

            if not grabbed1:
                print(f"Failed to grab from {CAMERA_1}")
                break

            if not grabbed2:
                print(f"Failed to grab from {CAMERA_2}")
                break

            ret1, frame1 = cap1.retrieve()
            ret2, frame2 = cap2.retrieve()

            if not ret1 or frame1 is None:
                print(f"Failed to retrieve from {CAMERA_1}")
                break

            if not ret2 or frame2 is None:
                print(f"Failed to retrieve from {CAMERA_2}")
                break

            # Reduce only the preview size.
            preview1 = cv2.resize(
                frame1,
                None,
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_AREA,
            )
            preview2 = cv2.resize(
                frame2,
                None,
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow("16MP Camera 1", preview1)
            cv2.imshow("16MP Camera 2", preview2)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                file1 = SAVE_DIR / f"{timestamp}_cam_1.jpg"
                file2 = SAVE_DIR / f"{timestamp}_cam_2.jpg"

                parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]

                success1 = cv2.imwrite(
                    str(file1),
                    frame1,
                    parameters,
                )
                success2 = cv2.imwrite(
                    str(file2),
                    frame2,
                    parameters,
                )

                print(
                    f"Camera 1 saved: {success1}, {file1}"
                )
                print(
                    f"Camera 2 saved: {success2}, {file2}"
                )

    except RuntimeError as error:
        print(error)

    finally:
        if cap1 is not None:
            cap1.release()

        if cap2 is not None:
            cap2.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
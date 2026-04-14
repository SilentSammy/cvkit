import cv2
import numpy as np
from video_source import VideoSource, CameraIntrinsics, CaptureSource, FileSource, controls

if __name__ == "__main__":
    import time

    dt = 0.0
    def on_read(source: VideoSource, frame=None):
        global dt
        dt = source.last_dt or dt
        controls(source, frame)

    # Set up VideoSource
    # cap = CaptureSource(0,
    # cap = CaptureSource("http://10.22.209.148:4747/video",
    cap = FileSource(r"capybara.mp4", loop=False,
    # cap = FileSource(r"resources\cameras\wide_angle_res2\calibration/*.*",
        auto_restart=True,
        on_read=on_read,
        intrinsics=CameraIntrinsics(
            K=np.array([[476.21413568, 0., 324.64535892], [0., 476.57490297, 242.01755433], [0., 0., 1.]], dtype=np.float32),
            # D=np.array([0.37628059, 0.8828322, -4.22102342, 5.72132593], dtype=np.float32),
            D=np.zeros(5),
            size=(640, 480)
        )
    )

    # cap = cv2.VideoCapture("http://10.22.209.148:4747/video")

    _last_t = None
    while True:
        t = time.time()
        dt = t - _last_t if _last_t is not None else None
        _last_t = t
        ret, frame = cap.read()

        if not ret:
            print("Waiting for source...", end="\r")
            cv2.waitKey(100)
            continue

        print(f"dt: {dt:.4f}s" if dt is not None else "dt: N/A")
        cv2.imshow('Camera', frame)
        # time.sleep(0.2) # Simulate processing delay
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

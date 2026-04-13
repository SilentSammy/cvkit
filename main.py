import cv2
import numpy as np
from video_source import VideoSource, CameraIntrinsics, CaptureSource, FileSource

if __name__ == "__main__":
    import time
    import user_input
    dt = 0.0
    
    def controls(source: VideoSource):
        global dt
        if source.last_dt is not None:
            dt = source.last_dt
        # Press to take a screenshot
        if user_input.rising_edge('t'):
            source.screenshot()
        
        # Press to toggle recording
        if user_input.rising_edge('r'):
            if source._recorder is None:
                source.start_recording()
            else:
                source.save_recording()
        
        # FileSource playback controls
        if isinstance(source, FileSource):
            if user_input.rising_edge('j'):
                source.play(-1.0)  # Reverse
            
            if user_input.rising_edge('k'):
                source.play(0.0)  # Stop
            
            if user_input.rising_edge('l'):
                source.play(1.0)  # Play
            
            if user_input.rising_edge(','):
                source.seek(-1)  # Seek backward 1 frame
            
            if user_input.rising_edge('.'):
                source.seek(1)  # Seek forward 1 frame

    # Open camera (index 0) with auto-restart on disconnect
    # cap = CaptureSource(0, auto_restart=True, on_read=controls)
    # Camera Matrix (K)

    # cap = CaptureSource(0,
    # cap = CaptureSource(r"capybara.mp4",
    cap = CaptureSource("http://192.168.137.2:4747/video",
    # cap = FileSource(r"capybara.mp4",
    # cap = FileSource(r"resources\cameras\wide_angle_res2\calibration/*.*",
        auto_restart=True,
        on_read=controls,
        intrinsics=CameraIntrinsics(
            K=np.array([[476.21413568, 0., 324.64535892], [0., 476.57490297, 242.01755433], [0., 0., 1.]], dtype=np.float32),
            # D=np.array([0.37628059, 0.8828322, -4.22102342, 5.72132593], dtype=np.float32),
            D=np.zeros(5),
            size=(640, 480)
        )
    )

    # cap = cv2.VideoCapture("http://192.168.137.2:4747/video")

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

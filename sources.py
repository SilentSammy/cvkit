from video_source import VideoSource, CameraIntrinsics, CaptureSource, FileSource, controls
import numpy as np

# webcam = CaptureSource(1,
#     # cap = FileSource(r"resources\cameras\wide_angle_res2\calibration/*.*",
#     auto_restart=True,
#     # on_read=on_read,
#     intrinsics=CameraIntrinsics(
#         K=np.array([[735.09668766, 0., 308.18011975], [0., 735.62248422, 242.58646203], [0., 0., 1.]], dtype=np.float32),
#         D=np.array([0.15017654, -1.34648531, 0.00405315, -0.00410719, 2.41472656], dtype=np.float32),
#         # D=np.zeros(5),
#         size=(640, 480)
#     )
# )

# wide_angle_3 = FileSource(r"resources\cameras\wide_angle_res3\test\*.*",
#     auto_restart=True,
#     # on_read=on_read,
#     intrinsics=CameraIntrinsics(
#         K=np.array([[793.9798621618975, 0.0, 628.3131432349588], [0.0, 793.3904503227144, 375.7912014522259], [0.0, 0.0, 1.0]], dtype=np.float32),
#         D=np.array([-0.3515292796708493, 0.158025188818097, -1.861499533667287e-05, -0.00031474130783931936, -0.03843522930855781], dtype=np.float32),
#         # D=np.zeros(5),
#         size=(1280, 720)
#     )
# )

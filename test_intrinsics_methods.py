import cv2
import numpy as np
import shutil
from video_source import CameraIntrinsics

# Create test intrinsics
intrinsics = CameraIntrinsics(
    K=np.array([[476.21413568, 0., 324.64535892], 
                [0., 476.57490297, 242.01755433], 
                [0., 0., 1.]], dtype=np.float32),
    D=np.array([0.37628059, 0.8828322, -4.22102342, 5.72132593], dtype=np.float32),
    size=(640, 480)
)

# Test files
sc_path = r"screenshots\screenshot_1776025790.png"
video_path = r"recordings\recording_1776025766.mp4"

print("=" * 60)
print("Testing CameraIntrinsics methods")
print("=" * 60)

# Test 1: append_to_filename
print("\n1. Testing append_to_filename()")
print("-" * 60)
sc_with_intrinsics = intrinsics.append_to_filename(sc_path.replace('.png', '_test.png'))
video_with_intrinsics = intrinsics.append_to_filename(video_path.replace('.mp4', '_test.mp4'))

shutil.copy(sc_path, sc_with_intrinsics)
shutil.copy(video_path, video_with_intrinsics)

print(f"PNG: {sc_with_intrinsics}")
print(f"MP4: {video_with_intrinsics}")

# Test 2: from_file with base64 filename
print("\n2. Testing from_file() with base64 filename")
print("-" * 60)
recovered_png = CameraIntrinsics.from_file(sc_with_intrinsics)
recovered_mp4 = CameraIntrinsics.from_file(video_with_intrinsics)

if recovered_png:
    print(f"✓ PNG: K match={np.allclose(recovered_png.K, intrinsics.K)}, "
          f"D match={np.allclose(recovered_png.D, intrinsics.D)}, "
          f"size match={recovered_png.size == intrinsics.size}")
else:
    print("✗ PNG: Failed to load")

if recovered_mp4:
    print(f"✓ MP4: K match={np.allclose(recovered_mp4.K, intrinsics.K)}, "
          f"D match={np.allclose(recovered_mp4.D, intrinsics.D)}, "
          f"size match={recovered_mp4.size == intrinsics.size}")
else:
    print("✗ MP4: Failed to load")

# Test 3: store_as_metadata
print("\n3. Testing store_as_metadata()")
print("-" * 60)
sc_metadata = sc_path.replace('.png', '_metadata.png')
video_metadata = video_path.replace('.mp4', '_metadata.mp4')

shutil.copy(sc_path, sc_metadata)
shutil.copy(video_path, video_metadata)

try:
    intrinsics.store_as_metadata(sc_metadata)
    print(f"✓ Stored PNG metadata: {sc_metadata}")
except Exception as e:
    print(f"✗ PNG metadata error: {e}")

try:
    intrinsics.store_as_metadata(video_metadata)
    print(f"✓ Stored MP4 metadata: {video_metadata}")
except Exception as e:
    print(f"✗ MP4 metadata error: {e}")

# Test 4: from_file with metadata
print("\n4. Testing from_file() with metadata")
print("-" * 60)
recovered_png_meta = CameraIntrinsics.from_file(sc_metadata)
recovered_mp4_meta = CameraIntrinsics.from_file(video_metadata)

if recovered_png_meta:
    print(f"✓ PNG: K match={np.allclose(recovered_png_meta.K, intrinsics.K)}, "
          f"D match={np.allclose(recovered_png_meta.D, intrinsics.D)}, "
          f"size match={recovered_png_meta.size == intrinsics.size}")
else:
    print("✗ PNG: Failed to load")

if recovered_mp4_meta:
    print(f"✓ MP4: K match={np.allclose(recovered_mp4_meta.K, intrinsics.K)}, "
          f"D match={np.allclose(recovered_mp4_meta.D, intrinsics.D)}, "
          f"size match={recovered_mp4_meta.size == intrinsics.size}")
else:
    print("✗ MP4: Failed to load")

print("\n" + "=" * 60)
print("All tests complete")
print("=" * 60)

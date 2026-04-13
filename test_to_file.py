import cv2
import numpy as np
import shutil
import os
from video_source import CameraIntrinsics

# Create test intrinsics
intrinsics = CameraIntrinsics(
    K=np.array([[476.21413568, 0., 324.64535892], 
                [0., 476.57490297, 242.01755433], 
                [0., 0., 1.]], dtype=np.float32),
    D=np.array([0.37628059, 0.8828322, -4.22102342, 5.72132593], dtype=np.float32),
    size=(640, 480)
)

print("=" * 60)
print("Testing CameraIntrinsics.to_file()")
print("=" * 60)

# Test 1: PNG with metadata (should succeed)
print("\n1. Testing PNG with Pillow available")
print("-" * 60)
sc_path = r"screenshots\screenshot_1776025790.png"
test_png = "test_tofile.png"
shutil.copy(sc_path, test_png)

result_path = intrinsics.to_file(test_png)
print(f"Input:  {test_png}")
print(f"Result: {result_path}")
print(f"Same path (metadata used): {result_path == test_png}")

# Verify it can be loaded back
recovered = CameraIntrinsics.from_file(result_path)
if recovered:
    print(f"✓ Round-trip successful: K match={np.allclose(recovered.K, intrinsics.K)}")
else:
    print("✗ Failed to load back")

os.remove(test_png)

# Test 2: MP4 with metadata (should succeed)
print("\n2. Testing MP4 with mutagen available")
print("-" * 60)
video_path = r"recordings\recording_1776025766.mp4"
test_mp4 = "test_tofile.mp4"
shutil.copy(video_path, test_mp4)

result_path = intrinsics.to_file(test_mp4)
print(f"Input:  {test_mp4}")
print(f"Result: {result_path}")
print(f"Same path (metadata used): {result_path == test_mp4}")

# Verify it can be loaded back
recovered = CameraIntrinsics.from_file(result_path)
if recovered:
    print(f"✓ Round-trip successful: K match={np.allclose(recovered.K, intrinsics.K)}")
else:
    print("✗ Failed to load back")

os.remove(test_mp4)

# Test 3: Unsupported format (should fall back to filename)
print("\n3. Testing unsupported format (.avi) - should fall back to filename")
print("-" * 60)
test_avi = "test_tofile.avi"
# Create a dummy file
with open(test_avi, 'w') as f:
    f.write("dummy")

result_path = intrinsics.to_file(test_avi)
print(f"Input:  {test_avi}")
print(f"Result: {result_path}")
print(f"Different path (filename encoding used): {result_path != test_avi}")
print(f"Has '__' separator: {'__' in result_path}")

# Verify it can be loaded back from the new filename
recovered = CameraIntrinsics.from_file(result_path)
if recovered:
    print(f"✓ Can decode from filename: K match={np.allclose(recovered.K, intrinsics.K)}")
else:
    print("✗ Failed to decode from filename")

os.remove(test_avi)

print("\n" + "=" * 60)
print("All to_file() tests complete")
print("=" * 60)

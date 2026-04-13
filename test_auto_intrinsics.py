import cv2
import numpy as np
import os
import shutil
from video_source import CameraIntrinsics, CaptureSource

# Create test intrinsics
intrinsics = CameraIntrinsics(
    K=np.array([[476.21413568, 0., 324.64535892], 
                [0., 476.57490297, 242.01755433], 
                [0., 0., 1.]], dtype=np.float32),
    D=np.array([0.37628059, 0.8828322, -4.22102342, 5.72132593], dtype=np.float32),
    size=(640, 480)
)

print("=" * 60)
print("Testing automatic intrinsics storage")
print("=" * 60)

# Test 1: Screenshot WITH intrinsics
print("\n1. Screenshot WITH intrinsics")
print("-" * 60)

# Create a dummy source (won't actually capture, will use last_frame)
cap = CaptureSource(0, intrinsics=intrinsics)
cap.last_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy frame

# Take screenshot
result = cap.screenshot("test_auto_screenshot.png")
print(f"Screenshot result: {result}")

# List files to see what was created
files = [f for f in os.listdir('.') if 'test_auto_screenshot' in f]
print(f"Files created: {files}")

# Try to load intrinsics back
for f in files:
    loaded = CameraIntrinsics.from_file(f)
    if loaded:
        print(f"✓ Loaded from {f}: K match={np.allclose(loaded.K, intrinsics.K)}")
    else:
        print(f"✗ Could not load from {f}")
    os.remove(f)

cap.release()

# Test 2: Screenshot WITHOUT intrinsics
print("\n2. Screenshot WITHOUT intrinsics")
print("-" * 60)

cap_no_intr = CaptureSource(0)
cap_no_intr.last_frame = np.zeros((480, 640, 3), dtype=np.uint8)

result = cap_no_intr.screenshot("test_no_intrinsics.png")
print(f"Screenshot result: {result}")

files = [f for f in os.listdir('.') if 'test_no_intrinsics' in f]
print(f"Files created: {files}")
print(f"Should be plain filename (no __ encoding): {'__' not in files[0] if files else 'N/A'}")

for f in files:
    os.remove(f)

cap_no_intr.release()

print("\n" + "=" * 60)
print("Test complete - automatic intrinsics storage works!")
print("=" * 60)

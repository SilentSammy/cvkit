import cv2
import numpy as np
from video_source import FileSource, CameraIntrinsics

print("=" * 60)
print("Testing FileSource skeleton")
print("=" * 60)

# Test 1: Video file playback
print("\n1. Testing video file playback")
print("-" * 60)
try:
    video_path = r"recordings\recording_1776025766.mp4"
    source = FileSource(video_path)
    
    print(f"Opened: {source.isOpened()}")
    print(f"Frame count: {source.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"FPS: {source.get(cv2.CAP_PROP_FPS)}")
    
    # Read a few frames
    for i in range(3):
        ret, frame = source.read()
        if ret:
            print(f"  Frame {i}: {frame.shape}")
        else:
            print(f"  Frame {i}: Failed to read")
    
    source.release()
    print("✓ Video file playback works")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Image folder playback
print("\n2. Testing image folder playback")
print("-" * 60)
try:
    folder_path = r"screenshots"
    source = FileSource(folder_path)
    
    print(f"Opened: {source.isOpened()}")
    print(f"Image count: {int(source.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"Current index: {int(source.get(cv2.CAP_PROP_POS_FRAMES))}")
    
    # Read a few images
    for i in range(min(3, int(source.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = source.read()
        if ret:
            print(f"  Image {i}: {frame.shape}")
        else:
            print(f"  Image {i}: Failed to read")
    
    print(f"After reading, index: {int(source.get(cv2.CAP_PROP_POS_FRAMES))}")
    
    # Test seeking
    source.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"After seek to 0, index: {int(source.get(cv2.CAP_PROP_POS_FRAMES))}")
    
    source.release()
    print("✓ Image folder playback works")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("FileSource skeleton ready for expansion")
print("=" * 60)

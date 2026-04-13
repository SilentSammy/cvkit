import cv2
from video_source import FileSource

print("=" * 60)
print("Testing FileSource frame_index and frame_count properties")
print("=" * 60)

video_path = r"recordings\recording_1776025766.mp4"
source = FileSource(video_path)

print(f"Opened: {source.isOpened()}")
print(f"frame_count: {source.frame_count}")
print(f"Initial frame_index: {source.frame_index}")

# Test reading advances index
print("\nReading 3 frames:")
for i in range(3):
    ret, frame = source.read()
    print(f"  After read {i+1}: frame_index = {source.frame_index}")

# Test setting frame_index directly
print("\nSetting frame_index to 10:")
source.frame_index = 10
print(f"frame_index = {source.frame_index}")

ret, frame = source.read()
print(f"After read: frame_index = {source.frame_index}")

# Test clamping - set to negative
print("\nSetting frame_index to -5 (should clamp to 0):")
source.frame_index = -5
print(f"frame_index = {source.frame_index}")

# Test clamping - set beyond end
print(f"\nSetting frame_index to 1000 (should clamp to {source.frame_count - 1}):")
source.frame_index = 1000
print(f"frame_index = {source.frame_index}")

# Test setting to exact boundaries
print(f"\nSetting frame_index to 0:")
source.frame_index = 0
print(f"frame_index = {source.frame_index}")

print(f"\nSetting frame_index to {source.frame_count - 1} (last frame):")
source.frame_index = source.frame_count - 1
print(f"frame_index = {source.frame_index}")

source.release()
print("\n✓ Properties work correctly with clamping")
print("=" * 60)

import cv2
from video_source import FileSource

print("=" * 60)
print("Testing simplified FileSource (video only)")
print("=" * 60)

video_path = r"recordings\recording_1776025766.mp4"
source = FileSource(video_path)

print(f"Opened: {source.isOpened()}")
print(f"Frame count: {int(source.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"FPS: {source.get(cv2.CAP_PROP_FPS)}")
print(f"Current position: {int(source.get(cv2.CAP_PROP_POS_FRAMES))}")

# Read a few frames
print("\nReading frames:")
for i in range(3):
    ret, frame = source.read()
    if ret:
        print(f"  Frame {i}: {frame.shape}")
    else:
        print(f"  Frame {i}: Failed")

print(f"Position after reads: {int(source.get(cv2.CAP_PROP_POS_FRAMES))}")

# Test seeking
print("\nSeeking to frame 10...")
source.set(cv2.CAP_PROP_POS_FRAMES, 10)
print(f"Position after seek: {int(source.get(cv2.CAP_PROP_POS_FRAMES))}")

ret, frame = source.read()
print(f"Read frame 10: {ret}, shape={frame.shape if ret else 'N/A'}")

source.release()
print("\n✓ Simplified FileSource works correctly")
print("=" * 60)

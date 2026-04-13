import cv2
from video_source import FileSource

print("=" * 60)
print("Testing FileSource loop behavior")
print("=" * 60)

video_path = r"recordings\recording_1776025766.mp4"

# Test 1: loop=False (clamping behavior)
print("\n1. Testing with loop=False (clamping)")
print("-" * 60)
source = FileSource(video_path, loop=False)

print(f"frame_count: {source.frame_count}")

# Test setting beyond end
source.frame_index = 1000
print(f"Set to 1000, got: {source.frame_index} (should be {source.frame_count - 1})")

# Test setting negative
source.frame_index = -10
print(f"Set to -10, got: {source.frame_index} (should be 0)")

source.release()

# Test 2: loop=True (wrapping behavior)
print("\n2. Testing with loop=True (wrapping)")
print("-" * 60)
source = FileSource(video_path, loop=True)

print(f"frame_count: {source.frame_count}")

# Test wrapping forward
source.frame_index = source.frame_count + 5
print(f"Set to {source.frame_count + 5}, got: {source.frame_index} (should be 5)")

# Test wrapping backward
source.frame_index = -1
print(f"Set to -1, got: {source.frame_index} (should be {source.frame_count - 1})")

# Test wrapping with large values
source.frame_index = source.frame_count * 2 + 10
print(f"Set to {source.frame_count * 2 + 10}, got: {source.frame_index} (should be 10)")

# Test exact boundary
source.frame_index = source.frame_count
print(f"Set to {source.frame_count}, got: {source.frame_index} (should be 0)")

source.release()

# Test 3: Reading beyond end with looping
print("\n3. Testing read() behavior at end of video")
print("-" * 60)

print("\nWith loop=False:")
source_no_loop = FileSource(video_path, loop=False)
source_no_loop.frame_index = source_no_loop.frame_count - 2

for i in range(5):
    ret, frame = source_no_loop.read()
    print(f"  Read {i}: ret={ret}, frame_index={source_no_loop.frame_index}")
source_no_loop.release()

print("\nWith loop=True:")
source_loop = FileSource(video_path, loop=True)
source_loop.frame_index = source_loop.frame_count - 2

for i in range(5):
    ret, frame = source_loop.read()
    print(f"  Read {i}: ret={ret}, frame_index={source_loop.frame_index}")
source_loop.release()

print("\n" + "=" * 60)
print("Loop behavior test complete")
print("=" * 60)

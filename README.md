# cvkit

Python computer vision toolkit with enhanced video source handling.

## Features

- **VideoSource Framework**: Drop-in replacement for `cv2.VideoCapture` with:
  - Live camera capture
  - Video file playback with frame-precise seeking
  - Image sequence support (wildcards or single images)
  - Built-in recording with background threading
  - Screenshot capture
  - Camera intrinsics storage (metadata or filename-encoded)
  - Auto-reconnect for network streams
  - Delta time tracking for CV applications

## Quick Start

```python
from video_source import CaptureSource, FileSource

# Camera or stream
cap = CaptureSource(0, auto_restart=True)

# Video or image sequence
cap = FileSource("video.mp4")
cap = FileSource("folder/*.jpg")

# Use like cv2.VideoCapture
ret, frame = cap.read()
cap.screenshot()
cap.start_recording()
```

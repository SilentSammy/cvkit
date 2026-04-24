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

## Using as a Submodule

**Add to a parent repo:**
```bash
git submodule add https://github.com/SilentSammy/cvkit.git cvkit
git commit -m "add cvkit submodule"
```

**Clone a parent repo that includes this submodule:**
```bash
git clone --recurse-submodules <parent-repo-url>
# or, if already cloned:
git submodule update --init
```

**Pull latest changes into the submodule:**
```bash
git submodule update --remote cvkit
git add cvkit
git commit -m "update cvkit submodule"
```

**Install dependencies:**
```bash
pip install -r cvkit/requirements.txt
```

---

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

from abc import ABC, abstractmethod
import threading
import time
import cv2
import numpy as np
try:
    import user_input
except ImportError:
    user_input = None


class CameraIntrinsics:
    """Camera calibration parameters with filename and metadata storage."""
    
    def __init__(self, K: np.ndarray, D: np.ndarray, size: tuple[int, int]):
        """Initialize camera intrinsics.
        
        Args:
            K: 3x3 camera matrix
            D: distortion coefficients
            size: (width, height) tuple
        """
        self.K = K
        self.D = D
        self.size = size
    
    def _to_base64(self) -> str:
        """Encode intrinsics as compact base64url string."""
        import base64
        import struct
        
        d_len = len(self.D)
        fmt = f'<9fB{d_len}f2H'
        data = struct.pack(fmt, *self.K.flatten(), d_len, *self.D, *self.size)
        return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')
    
    @staticmethod
    def _from_base64(encoded: str) -> 'CameraIntrinsics':
        """Decode intrinsics from base64url string."""
        import base64
        import struct
        
        # Add padding if needed
        padding = (4 - len(encoded) % 4) % 4
        encoded += '=' * padding
        data = base64.urlsafe_b64decode(encoded)
        
        # Unpack K (9 floats = 36 bytes)
        K_flat = struct.unpack('<9f', data[:36])
        K = np.array(K_flat, dtype=np.float32).reshape(3, 3)
        
        # Get D length (1 byte)
        d_len = struct.unpack('<B', data[36:37])[0]
        
        # Unpack D
        offset = 37
        D = np.array(struct.unpack(f'<{d_len}f', data[offset:offset+d_len*4]), dtype=np.float32)
        offset += d_len * 4
        
        # Unpack size (2 unsigned shorts)
        size = struct.unpack('<2H', data[offset:offset+4])
        
        return CameraIntrinsics(K=K, D=D, size=size)
    
    def append_to_filename(self, filepath: str) -> str:
        """Return a new filepath with base64-encoded intrinsics appended to filename.
        
        Args:
            filepath: Original file path
            
        Returns:
            New filepath with intrinsics encoded in filename
            
        Example:
            "video.mp4" -> "video__aRvuQwAAAACb...mp4"
        """
        import os
        base, ext = os.path.splitext(filepath)
        encoded = self._to_base64()
        return f"{base}__{encoded}{ext}"
    
    def store_as_metadata(self, filepath: str):
        """Store intrinsics in file metadata (PNG tEXt or MP4 comment field).
        
        Args:
            filepath: Path to PNG or MP4 file
            
        Raises:
            ImportError: If Pillow (PNG) or mutagen (MP4) is not installed
            ValueError: If file type is not PNG or MP4
        """
        import json
        import os
        
        ext = os.path.splitext(filepath)[1].lower()
        
        intrinsics_dict = {
            'K': self.K.tolist(),
            'D': self.D.tolist(),
            'size': self.size
        }
        
        if ext == '.png':
            from PIL import Image, PngImagePlugin
            
            img = Image.open(filepath)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("camera_intrinsics", json.dumps(intrinsics_dict))
            img.save(filepath, pnginfo=metadata)
            
        elif ext == '.mp4':
            from mutagen.mp4 import MP4
            
            video = MP4(filepath)
            video["\xa9cmt"] = json.dumps(intrinsics_dict)
            video.save()
            
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .png or .mp4")
    
    def to_file(self, filepath: str) -> str:
        """Store intrinsics using metadata if possible, otherwise encode in filename.
        
        Tries metadata storage first (requires Pillow/mutagen). If that fails or the
        file type is unsupported, returns a new filepath with base64-encoded intrinsics.
        
        Args:
            filepath: Path to file
            
        Returns:
            Original filepath if metadata was stored successfully, otherwise a new
            filepath with base64-encoded intrinsics in the filename
            
        Note:
            When filename encoding is used, the caller must rename the file to the
            returned path. The metadata approach modifies the file in-place.
        """
        try:
            self.store_as_metadata(filepath)
            return filepath
        except (ImportError, ValueError):
            # Fall back to filename encoding
            return self.append_to_filename(filepath)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'CameraIntrinsics | None':
        """Load intrinsics from file (tries base64 filename first, then metadata).
        
        Args:
            filepath: Path to file with embedded intrinsics
            
        Returns:
            CameraIntrinsics if found, None otherwise
            
        Raises:
            ImportError: If metadata loading requires missing Pillow/mutagen
        """
        import os
        import json
        
        # Try base64 from filename first
        base, ext = os.path.splitext(filepath)
        if '__' in base:
            parts = base.rsplit('__', 1)
            if len(parts) == 2:
                try:
                    return cls._from_base64(parts[1])
                except Exception:
                    pass
        
        # Try metadata (lazy import)
        ext = ext.lower()
        
        if ext == '.png':
            from PIL import Image
            
            img = Image.open(filepath)
            if "camera_intrinsics" in img.text:
                data = json.loads(img.text["camera_intrinsics"])
                return cls(
                    K=np.array(data['K'], dtype=np.float32),
                    D=np.array(data['D'], dtype=np.float32),
                    size=tuple(data['size'])
                )
                
        elif ext == '.mp4':
            from mutagen.mp4 import MP4
            
            video = MP4(filepath)
            if "\xa9cmt" in video:
                data = json.loads(video["\xa9cmt"][0])
                return cls(
                    K=np.array(data['K'], dtype=np.float32),
                    D=np.array(data['D'], dtype=np.float32),
                    size=tuple(data['size'])
                )
        
        return None


class VideoSource(ABC):
    """Base class for all video sources, mirroring the cv2.VideoCapture API."""

    def __init__(self, source, name: str = "", auto_restart: bool = False, on_read=None,
                 screenshot_name=None, recording_name=None, intrinsics: CameraIntrinsics | None = None):
        """Initialize and open the source."""
        self._source = source
        self.name = name
        self._intrinsics: CameraIntrinsics | None = intrinsics
        self.last_frame = None
        self.last_dt = None
        self._last_read_timestamp = None
        self._auto_restart = auto_restart
        self._on_read = on_read or controls
        # Set default name generators
        if screenshot_name is None:
            screenshot_name = lambda: f"screenshots/{self.name}_screenshot_{int(time.time())}.png" if self.name else f"screenshots/screenshot_{int(time.time())}.png"
        if recording_name is None:
            recording_name = lambda: f"recordings/{self.name}_recording_{int(time.time())}.mp4" if self.name else f"recordings/recording_{int(time.time())}.mp4"
        self._screenshot_name = screenshot_name
        self._recording_name = recording_name
        self._recorder = None
        self._recording_path = None
        self._lock = threading.Lock()
        self._reconnecting = False
        self._stop_event = threading.Event()
        self._reconnect_thread = None
        self._recording_thread = None
        self._recording_stop_event = threading.Event()
        self._start(source)

    def read(self) -> tuple[bool, object]:
        """Return the next frame. Always fires on_read; last_frame holds the most recent valid frame."""
        current_time = time.time()
        prev_timestamp = self._last_read_timestamp
        self._last_read_timestamp = current_time

        if self._reconnecting:
            if prev_timestamp is not None:
                self.last_dt = current_time - prev_timestamp
            if self._on_read:
                self._on_read(self, None)
            return False, None
        with self._lock:
            ret, frame = self._read()
        self.last_dt = self._get_dt(current_time, prev_timestamp)
        if ret:
            self.last_frame = frame
        if not ret and self._auto_restart and not self._reconnecting:
            self._reconnecting = True
            self._reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
            self._reconnect_thread.start()
        if self._on_read:
            self._on_read(self, frame)
        return ret, frame

    def _get_dt(self, current_time: float, prev_timestamp: float | None) -> float | None:
        """Compute last_dt for this read. Override to customize dt reporting."""
        if prev_timestamp is None:
            return None
        return current_time - prev_timestamp

    def _reconnect_loop(self):
        """Background thread: repeatedly attempt _stop/_start until the source is open again."""
        while not self._stop_event.is_set():
            with self._lock:
                try:
                    self._stop()
                    self._start(self._source)
                except Exception:
                    pass
            if self._is_opened():
                self._reconnecting = False
                return
            self._stop_event.wait(timeout=2.0)

    def isOpened(self) -> bool:
        """Return True if the source is currently open."""
        return self._is_opened()

    def get_intrinsics(self) -> CameraIntrinsics | None:
        """Return camera intrinsics for the current frame. Override if intrinsics vary per frame."""
        return self._intrinsics

    def screenshot(self, path: str = None) -> bool:
        """Save the current or last known frame to a file. Returns True on success.
        
        If intrinsics are available, they will be automatically stored in the file
        (using metadata if possible, otherwise encoded in the filename).
        """
        if path is None:
            path = self._screenshot_name()
        ret, frame = self.read()
        frame = frame if ret else self.last_frame
        if frame is not None:
            import os
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            cv2.imwrite(path, frame)
            
            # Store intrinsics if available
            if self._intrinsics is not None:
                new_path = self._intrinsics.to_file(path)
                if new_path != path:
                    # Filename encoding was used, rename the file
                    os.rename(path, new_path)
                    path = new_path
            
            print(f"Screenshot saved: {path}")
            return True
        return False

    def _recording_loop(self, fps: float):
        """Background thread: write last_frame to recorder at fixed fps."""
        interval = 1.0 / fps
        while not self._recording_stop_event.is_set():
            start_time = time.time()
            
            with self._lock:
                if self.last_frame is not None and self._recorder is not None:
                    self._recorder.write(self.last_frame)
            
            # Sleep for the remaining time to maintain fps
            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._recording_stop_event.wait(timeout=sleep_time)

    def start_recording(self, path: str = None, fps: float = None, size: tuple[int, int] = None) -> bool:
        """Begin recording frames. Auto-detects fps and size if not provided. Returns True on success."""
        if self._recorder is not None:
            return False  # Already recording
        
        # Auto-detect fps from source
        if fps is None:
            fps = self.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
        
        # Auto-detect size from last_frame
        if size is None:
            if self.last_frame is not None:
                h, w = self.last_frame.shape[:2]
                size = (w, h)
            else:
                size = (640, 480)  # fallback
        
        # Use temp path if none provided
        if path is None:
            path = f".temp_recording_{int(time.time())}.mp4"
        
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        self._recording_path = path
        
        # Try codecs in order of preference
        codecs_to_try = ['H264', 'avc1', 'mp4v', 'MJPG']
        for codec_str in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            # Use MSMF backend on Windows to avoid OpenH264 warnings
            import platform
            if platform.system() == 'Windows':
                self._recorder = cv2.VideoWriter(path, cv2.CAP_MSMF, fourcc, fps, size)
            else:
                self._recorder = cv2.VideoWriter(path, fourcc, fps, size)
            if self._recorder.isOpened():
                # Start background recording thread
                self._recording_stop_event.clear()
                self._recording_thread = threading.Thread(target=self._recording_loop, args=(fps,), daemon=True)
                self._recording_thread.start()
                print(f"Recording started: {path} (codec: {codec_str}, fps: {fps})")
                return True
        
        # All codecs failed
        self._recorder = None
        self._recording_path = None
        return False

    def save_recording(self, path: str = None) -> bool:
        """Finalize recording. If path is provided, rename before closing. Returns True on success.
        
        If intrinsics are available, they will be automatically stored in the file
        (using metadata if possible, otherwise encoded in the filename).
        """
        if self._recorder is None:
            return False
        
        # Stop background recording thread
        self._recording_stop_event.set()
        if self._recording_thread is not None:
            self._recording_thread.join(timeout=1.0)
            self._recording_thread = None
        
        self._recorder.release()
        self._recorder = None
        
        # Rename if a new path is requested
        if path is None:
            path = self._recording_name()
        
        if path != self._recording_path:
            import os
            try:
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                os.rename(self._recording_path, path)
            except Exception as e:
                print(f"Failed to rename recording: {e}")
                self._recording_path = None
                return False
        
        # Store intrinsics if available
        if self._intrinsics is not None:
            import os
            new_path = self._intrinsics.to_file(path)
            if new_path != path:
                # Filename encoding was used, rename the file
                os.rename(path, new_path)
                path = new_path
        
        print(f"Recording saved: {path}")
        self._recording_path = None
        return True
# Stop background recording thread
        self._recording_stop_event.set()
        if self._recording_thread is not None:
            self._recording_thread.join(timeout=1.0)
            self._recording_thread = None
         
    def stop_recording(self) -> bool:
        """Discard the current recording without saving."""
        if self._recorder is None:
            return False
        
        self._recorder.release()
        self._recorder = None
        
        import os
        if self._recording_path and os.path.exists(self._recording_path):
            try:
                os.remove(self._recording_path)
                print(f"Recording discarded: {self._recording_path}")
            except Exception:
                pass
        
        self._recording_path = None
        return True

    def release(self):
        """Signal the reconnect thread to stop, then release the source."""
        self._stop_event.set()
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=3.0)
        if self._recorder is not None:
            self.stop_recording()  # Discard unsaved recording
        with self._lock:
            self._stop()

    def get(self, prop_id: int) -> float:
        """Get a source property (cv2.CAP_PROP_*)."""
        return self._get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        """Set a source property (cv2.CAP_PROP_*)."""
        return self._set(prop_id, value)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    @abstractmethod
    def _start(self, source): ...

    @abstractmethod
    def _stop(self): ...

    @abstractmethod
    def _read(self) -> tuple[bool, object]: ...

    @abstractmethod
    def _is_opened(self) -> bool: ...

    @abstractmethod
    def _get(self, prop_id: int) -> float: ...

    @abstractmethod
    def _set(self, prop_id: int, value: float) -> bool: ...


class CaptureSource(VideoSource):
    """VideoSource backed by cv2.VideoCapture — handles webcams, MJPEG streams, and any cv2-compatible source."""

    def _start(self, source):
        # Use FFMPEG backend for network streams to prevent fallback to CAP_IMAGES
        if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://') or source.startswith('rtsp://')):
            self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Slot for the latest decoded frame — written by grab thread, read by main thread
        self._cap_lock = threading.Lock()
        self._latest_frame = (False, None)
        self._frame_ready = threading.Event()
        # Per-start stop event so old grab threads exit cleanly on reconnect
        self._grab_stop = threading.Event()
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def _grab_loop(self):
        """Background thread: decode frames as fast as the source delivers them.
        
        Holds _cap_lock for the entire read() call to satisfy FFmpeg's internal
        thread-safety requirements. _read() on the main thread only touches the
        slot, so the main loop is never blocked waiting here.
        """
        while not self._stop_event.is_set() and not self._grab_stop.is_set():
            try:
                with self._cap_lock:
                    if not self._cap.isOpened():
                        return
                    ret, frame = self._cap.read()
                    # Copy the frame so _latest_frame owns its memory independently.
                    # FFmpeg reuses internal decode buffers on the next read(), which
                    # would corrupt the array we already stored without this copy.
                    self._latest_frame = (ret, frame.copy() if ret and frame is not None else frame)
                    if ret:
                        self._frame_ready.set()
            except Exception:
                self._latest_frame = (False, None)
                return
            if not ret:
                return

    def _stop(self):
        self._grab_stop.set()
        self._cap.release()

    def _read(self):
        # Block until the grab thread delivers its first frame, just like a plain
        # cap.read() would. This prevents auto_restart from seeing (False, None)
        # before the source has had a chance to initialize.
        if not self._frame_ready.is_set():
            self._frame_ready.wait(timeout=10.0)
        # No lock needed: tuple assignment is atomic under the GIL.
        # _cap_lock is held by _grab_loop for all _cap calls, keeping FFmpeg happy.
        return self._latest_frame

    def _is_opened(self):
        with self._cap_lock:
            return self._cap.isOpened()

    def _get(self, prop_id):
        with self._cap_lock:
            return self._cap.get(prop_id)

    def _set(self, prop_id, value):
        with self._cap_lock:
            return self._cap.set(prop_id, value)


class FileSource(VideoSource):
    """VideoSource for playback of video files."""
    
    @staticmethod
    def get_files(pattern: str) -> list[str]:
        """Get all files matching a wildcard pattern.
        
        Args:
            pattern: Wildcard pattern (e.g., "screenshots/*.png", "data/**/*.jpg")
        
        Returns:
            Sorted list of file paths matching the pattern
        
        Example:
            files = FileSource.get_files("screenshots/*.png")
        """
        import glob
        return sorted(glob.glob(pattern, recursive=True))
    
    def __init__(self, source, loop: bool = False, **kwargs):
        """Initialize FileSource.
        
        Args:
            source: Path to video file or wildcard pattern for image sequence
            loop: If True, frame_index wraps around; if False, it clamps to bounds
            **kwargs: Additional arguments passed to VideoSource
        """
        self.loop = loop
        self._play_speed = 0.0  # 0 = paused, 1.0 = normal speed, -1.0 = reverse, etc.
        self._last_read_time = None
        self._frame_index = 0.0  # Single source of truth for position (float for sub-frame precision)
        
        # Image sequence state
        self._is_image_sequence = False
        self._image_files = []
        
        # Intrinsics loaded from file
        self._retrieved_intrinsics: CameraIntrinsics | None = None
        self._last_reported_frame_index: int | None = None
        
        super().__init__(source, **kwargs)
    
    def _start(self, source):
        """Open video file or image sequence."""
        # Try to open as video first
        self._is_image_sequence = False
        self._cap = cv2.VideoCapture(source)
        
        if not self._cap.isOpened():
            # Failed to open as video — treat as image file or pattern
            self._is_image_sequence = True
            self._image_files = self.get_files(source)
        else:
            # Video file mode — attempt to load intrinsics
            if isinstance(source, str):
                try:
                    self._retrieved_intrinsics = CameraIntrinsics.from_file(source)
                except Exception:
                    self._retrieved_intrinsics = None
    
    def _stop(self):
        """Close the video file or image sequence."""
        if self._is_image_sequence:
            self._image_files = []
        else:
            self._cap.release()
    
    def _get_current_frame(self) -> tuple[bool, object]:
        """Get the frame at the current _frame_index position."""
        if self._is_image_sequence:
            # Image mode: load from file list
            idx = int(self._frame_index)
            if 0 <= idx < len(self._image_files):
                filepath = self._image_files[idx]
                frame = cv2.imread(filepath)
                
                # Try to load intrinsics from this image file
                try:
                    self._retrieved_intrinsics = CameraIntrinsics.from_file(filepath)
                except Exception:
                    self._retrieved_intrinsics = None
                
                return (frame is not None, frame)
            return (False, None)
        else:
            # Video mode: sync OpenCV to our position, then read
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_index)
            ret, frame = self._cap.read()
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_index)  # Restore (read advanced it)
            return ret, frame
    
    def get_intrinsics(self) -> CameraIntrinsics | None:
        """Return intrinsics loaded from the current file, or fallback to provided intrinsics."""
        return self._retrieved_intrinsics or self._intrinsics

    def _read(self) -> tuple[bool, object]:
        """Read current frame, with optional auto-advance based on play speed."""
        # Auto-advance based on elapsed time and play speed
        if self._play_speed != 0.0:
            current_time = time.time()
            
            if self._last_read_time is not None:
                elapsed = current_time - self._last_read_time
                fps = self._get(cv2.CAP_PROP_FPS)
                frame_delta = elapsed * fps * self._play_speed
                self.frame_index = self._frame_index + frame_delta
            
            self._last_read_time = current_time
        
        return self._get_current_frame()

    def _get_dt(self, current_time: float, prev_timestamp: float | None) -> float | None:
        """Use frame-based dt when the frame index changed since the last call, zero otherwise.
        
        Returns negative dt when playing in reverse to preserve directional information.
        """
        current_index = int(self._frame_index)
        if self._last_reported_frame_index is None:
            self._last_reported_frame_index = current_index
            return None
        frames_advanced = current_index - self._last_reported_frame_index
        self._last_reported_frame_index = current_index
        if frames_advanced != 0:
            fps = self._get(cv2.CAP_PROP_FPS)
            if fps > 0:
                return frames_advanced / fps
        return 0.0

    def _is_opened(self) -> bool:
        """Check if video file or image sequence is open."""
        if self._is_image_sequence:
            return len(self._image_files) > 0
        else:
            return self._cap.isOpened()
    
    def _get(self, prop_id: int) -> float:
        """Get video or image sequence property."""
        if self._is_image_sequence:
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                return self._frame_index
            elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return float(len(self._image_files))
            elif prop_id == cv2.CAP_PROP_FPS:
                return 30.0  # Default FPS for image sequences
            else:
                return 0.0
        else:
            return self._cap.get(prop_id)
    
    def _set(self, prop_id: int, value: float) -> bool:
        """Set video or image sequence property."""
        if self._is_image_sequence:
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self._frame_index = value
                return True
            return False
        else:
            return self._cap.set(prop_id, value)
    
    def seek(self, i: int):
        """Move frame position by i frames (positive or negative)."""
        self.frame_index = self.frame_index + i
    
    def play(self, speed: float):
        """Set playback speed for auto-advance.
        
        Args:
            speed: Playback speed multiplier
                   0.0 = paused (manual control)
                   1.0 = normal speed forward
                  -1.0 = normal speed reverse
                   2.0 = double speed forward
        """
        self._play_speed = speed
        self._last_read_time = None  # Reset timing on speed change
    
    @property
    def frame_index(self) -> int:
        """Current frame index (0-based). Returns nearest integer."""
        return int(self._frame_index)
    
    @frame_index.setter
    def frame_index(self, value: float | int):
        """Set current frame index. Accepts float for sub-frame precision."""
        if self.frame_count == 0:
            return
        
        # Update internal position with wrap/clamp logic
        if self.loop:
            self._frame_index = float(value) % self.frame_count
        else:
            self._frame_index = max(0.0, min(float(value), float(self.frame_count - 1)))
    
    @property
    def frame_count(self) -> int:
        """Total number of frames in the video or image sequence."""
        return int(self._get(cv2.CAP_PROP_FRAME_COUNT))


play_ctrls = {
    'j': -1.0,
    'k': 0.0,
    'l': 1.0,
}
seek_ctrls = {
    ',': -1,
    '.': 1,
}
record_ctrls = ('t', 'r')

def controls(source: VideoSource, frame=None):
    if user_input is None:
        return

    for key, action in zip(record_ctrls, (source.screenshot, lambda: source.start_recording() if source._recorder is None else source.save_recording())):
        if user_input.rising_edge(key):
            action()

    if isinstance(source, FileSource):
        for key, speed in play_ctrls.items():
            if user_input.rising_edge(key):
                source.play(speed)

        for key, delta in seek_ctrls.items():
            if user_input.rising_edge(key):
                source.seek(delta)

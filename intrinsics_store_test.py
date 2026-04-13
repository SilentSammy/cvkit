import cv2
import numpy as np
import json
import shutil
from video_source import VideoSource, CameraIntrinsics, CaptureSource

intrinsics=CameraIntrinsics(
    K=np.array([[476.21413568, 0., 324.64535892], [0., 476.57490297, 242.01755433], [0., 0., 1.]], dtype=np.float32),
    D=np.array([0.37628059, 0.8828322, -4.22102342, 5.72132593], dtype=np.float32),
    # D=np.zeros(5),
    size=(640, 480)
)

sc_path = r"screenshots\screenshot_1776025790.png"
video_path = r"recordings\recording_1776025766.mp4"

# Helper functions for serialization
def intrinsics_to_dict(intr: CameraIntrinsics) -> dict:
    """Convert CameraIntrinsics to a JSON-serializable dict."""
    return {
        'K': intr.K.tolist(),
        'D': intr.D.tolist(),
        'size': intr.size
    }

def dict_to_intrinsics(data: dict) -> CameraIntrinsics:
    """Convert dict back to CameraIntrinsics."""
    return CameraIntrinsics(
        K=np.array(data['K'], dtype=np.float32),
        D=np.array(data['D'], dtype=np.float32),
        size=tuple(data['size'])
    )

# ========== OPTION 1: PNG metadata with Pillow ==========
print("Testing Option 1: PNG metadata with Pillow")
try:
    from PIL import Image, PngImagePlugin
    
    # Make a copy
    sc_path_copy = sc_path.replace('.png', '_with_metadata.png')
    shutil.copy(sc_path, sc_path_copy)
    
    # Read the image with Pillow
    img = Image.open(sc_path_copy)
    
    # Create PNG metadata
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("camera_intrinsics", json.dumps(intrinsics_to_dict(intrinsics)))
    
    # Save with metadata
    img.save(sc_path_copy, pnginfo=metadata)
    print(f"✓ Saved PNG with metadata: {sc_path_copy}")
    
    # Read back and verify
    img_read = Image.open(sc_path_copy)
    if "camera_intrinsics" in img_read.text:
        recovered = dict_to_intrinsics(json.loads(img_read.text["camera_intrinsics"]))
        print(f"✓ Read back intrinsics from PNG")
        print(f"  K matches: {np.allclose(recovered.K, intrinsics.K)}")
        print(f"  D matches: {np.allclose(recovered.D, intrinsics.D)}")
        print(f"  size matches: {recovered.size == intrinsics.size}")
    else:
        print("✗ Failed to read metadata from PNG")
        
except ImportError:
    print("✗ Pillow not installed (pip install Pillow)")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# ========== OPTION 1: MP4 metadata with mutagen ==========
print("Testing Option 1: MP4 metadata with mutagen")
try:
    from mutagen.mp4 import MP4
    
    # Make a copy
    video_path_copy = video_path.replace('.mp4', '_with_metadata.mp4')
    shutil.copy(video_path, video_path_copy)
    
    # Open with mutagen
    video = MP4(video_path_copy)
    
    # Store in freeform metadata (using the '----' atom format)
    # Format: ----:mean:name = value
    video["\xa9cmt"] = json.dumps(intrinsics_to_dict(intrinsics))  # Use comment field
    video.save()
    print(f"✓ Saved MP4 with metadata: {video_path_copy}")
    
    # Read back and verify
    video_read = MP4(video_path_copy)
    if "\xa9cmt" in video_read:
        recovered = dict_to_intrinsics(json.loads(video_read["\xa9cmt"][0]))
        print(f"✓ Read back intrinsics from MP4")
        print(f"  K matches: {np.allclose(recovered.K, intrinsics.K)}")
        print(f"  D matches: {np.allclose(recovered.D, intrinsics.D)}")
        print(f"  size matches: {recovered.size == intrinsics.size}")
    else:
        print("✗ Failed to read metadata from MP4")
        
except ImportError:
    print("✗ mutagen not installed (pip install mutagen)")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# ========== OPTION 2: Base64 in filename ==========
print("Testing Option 2: Base64 encoded in filename")
try:
    import base64
    import struct
    import os
    
    def intrinsics_to_base64(intr: CameraIntrinsics) -> str:
        """Encode intrinsics as compact base64url string."""
        # Pack: 9 floats (K) + 1 uint8 (D length) + D floats + 2 uint16 (size)
        d_len = len(intr.D)
        # Use struct with explicit format: < = little-endian, f = float, B = unsigned byte, H = unsigned short
        fmt = f'<9fB{d_len}f2H'
        data = struct.pack(fmt, *intr.K.flatten(), d_len, *intr.D, *intr.size)
        return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')
    
    def base64_to_intrinsics(encoded: str) -> CameraIntrinsics:
        """Decode intrinsics from base64url string."""
        # Add padding if needed
        padding = (4 - len(encoded) % 4) % 4
        encoded += '=' * padding
        data = base64.urlsafe_b64decode(encoded)
        
        # Unpack K first (9 floats = 36 bytes)
        K_flat = struct.unpack('<9f', data[:36])
        K = np.array(K_flat, dtype=np.float32).reshape(3, 3)
        
        # Get D length (1 byte)
        d_len = struct.unpack('<B', data[36:37])[0]
        
        # Unpack D (d_len floats)
        offset = 37
        D = np.array(struct.unpack(f'<{d_len}f', data[offset:offset+d_len*4]), dtype=np.float32)
        offset += d_len * 4
        
        # Unpack size (2 unsigned shorts)
        size = struct.unpack('<2H', data[offset:offset+4])
        
        return CameraIntrinsics(K=K, D=D, size=size)
    
    def add_intrinsics_to_path(path: str, intr: CameraIntrinsics) -> str:
        """Add base64-encoded intrinsics to filename before extension."""
        base, ext = os.path.splitext(path)
        encoded = intrinsics_to_base64(intr)
        return f"{base}__{encoded}{ext}"
    
    def extract_intrinsics_from_path(path: str) -> tuple[str, CameraIntrinsics | None]:
        """Extract intrinsics from filename, return (clean_path, intrinsics)."""
        base, ext = os.path.splitext(path)
        if '__' in base:
            parts = base.rsplit('__', 1)
            if len(parts) == 2:
                clean_base, encoded = parts
                try:
                    intr = base64_to_intrinsics(encoded)
                    return f"{clean_base}{ext}", intr
                except Exception:
                    pass
        return path, None
    
    # Test PNG
    sc_path_b64 = add_intrinsics_to_path(sc_path.replace('.png', '_b64.png'), intrinsics)
    shutil.copy(sc_path, sc_path_b64)
    print(f"✓ Created PNG with encoded filename: {os.path.basename(sc_path_b64)}")
    print(f"  Filename length: {len(os.path.basename(sc_path_b64))} chars")
    
    clean_path, recovered = extract_intrinsics_from_path(sc_path_b64)
    if recovered:
        print(f"✓ Decoded intrinsics from PNG filename")
        print(f"  K matches: {np.allclose(recovered.K, intrinsics.K)}")
        print(f"  D matches: {np.allclose(recovered.D, intrinsics.D)}")
        print(f"  size matches: {recovered.size == intrinsics.size}")
    else:
        print("✗ Failed to decode from PNG filename")
    
    # Test MP4
    video_path_b64 = add_intrinsics_to_path(video_path.replace('.mp4', '_b64.mp4'), intrinsics)
    shutil.copy(video_path, video_path_b64)
    print(f"✓ Created MP4 with encoded filename: {os.path.basename(video_path_b64)}")
    print(f"  Filename length: {len(os.path.basename(video_path_b64))} chars")
    
    clean_path, recovered = extract_intrinsics_from_path(video_path_b64)
    if recovered:
        print(f"✓ Decoded intrinsics from MP4 filename")
        print(f"  K matches: {np.allclose(recovered.K, intrinsics.K)}")
        print(f"  D matches: {np.allclose(recovered.D, intrinsics.D)}")
        print(f"  size matches: {recovered.size == intrinsics.size}")
    else:
        print("✗ Failed to decode from MP4 filename")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""
ArUco Marker Detection and Camera Pose Visualization
Detects 4x4 ArUco markers and visualizes camera pose in 3D
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import argparse


def load_calibration(calibration_file):
    """Load camera calibration from file."""
    with open(calibration_file, 'r') as f:
        content = f.read()
    
    # Extract camera matrix
    K_start = content.find('K = np.array([')
    K_end = content.find('])', K_start) + 2
    K_str = content[K_start:K_end].replace('K = ', '')
    K = eval(K_str)
    
    # Extract distortion coefficients
    D_start = content.find('D = np.array(')
    D_end = content.find(')', D_start) + 1
    D_str = content[D_start:D_end].replace('D = ', '')
    D = eval(D_str)
    
    return np.array(K), np.array(D)


def detect_aruco_markers(image, aruco_dict, parameters):
    """Detect ArUco markers in image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try new API first (OpenCV 4.7+)
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        # Fall back to old API
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    return corners, ids, rejected


def estimate_pose(corners, marker_size, camera_matrix, dist_coeffs):
    """
    Estimate pose of ArUco marker.
    Returns rotation vector, translation vector
    """
    # Define 3D coordinates of marker corners in marker's coordinate system
    # Marker is centered at origin, lying in XY plane
    half_size = marker_size / 2
    obj_points = np.array([
        [-half_size,  half_size, 0],  # Top-left
        [ half_size,  half_size, 0],  # Top-right
        [ half_size, -half_size, 0],  # Bottom-right
        [-half_size, -half_size, 0]   # Bottom-left
    ], dtype=np.float32)
    
    # Estimate pose
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    return rvec, tvec if success else (None, None)


def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.05):
    """Draw 3D axis on the marker."""
    axis_points = np.float32([
        [0, 0, 0],           # Origin
        [length, 0, 0],      # X-axis (red)
        [0, length, 0],      # Y-axis (green)
        [0, 0, -length]      # Z-axis (blue)
    ])
    
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.astype(int)
    
    origin = tuple(img_points[0].ravel())
    img = cv2.line(img, origin, tuple(img_points[1].ravel()), (0, 0, 255), 3)  # X: Red
    img = cv2.line(img, origin, tuple(img_points[2].ravel()), (0, 255, 0), 3)  # Y: Green
    img = cv2.line(img, origin, tuple(img_points[3].ravel()), (255, 0, 0), 3)  # Z: Blue
    
    return img


def plot_camera_pose_3d(rvec, tvec, marker_size=0.1):
    """Plot camera pose in 3D using matplotlib."""
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Camera position in world coordinates
    # tvec is the translation from camera to marker
    # So camera position is -R.T @ tvec
    camera_pos = -R.T @ tvec.reshape(3, 1)
    
    # Camera orientation axes
    camera_axes_length = 0.05
    camera_x = R.T @ np.array([[camera_axes_length], [0], [0]]) + camera_pos
    camera_y = R.T @ np.array([[0], [camera_axes_length], [0]]) + camera_pos
    camera_z = R.T @ np.array([[0], [0], [camera_axes_length]]) + camera_pos
    
    # Marker corners in world coordinates (marker at origin)
    half = marker_size / 2
    marker_corners = np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0],
        [-half, -half, 0]  # Close the square
    ]).T
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot marker boundary
    ax.plot(marker_corners[0], marker_corners[1], marker_corners[2], 
            'b-', linewidth=2, label='ArUco Marker')
    
    # Plot marker as a surface instead of fill
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [list(zip(marker_corners[0][:-1], marker_corners[1][:-1], marker_corners[2][:-1]))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, facecolor='blue', edgecolor='blue'))
    
    # Plot world coordinate axes at marker center
    axis_len = marker_size
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2, label='World X')
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.3, linewidth=2, label='World Y')
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.3, linewidth=2, label='World Z')
    
    # Plot camera position
    ax.scatter(*camera_pos, color='black', s=100, marker='o', label='Camera')
    
    # Plot camera orientation axes
    ax.plot([camera_pos[0, 0], camera_x[0, 0]], 
            [camera_pos[1, 0], camera_x[1, 0]], 
            [camera_pos[2, 0], camera_x[2, 0]], 'r-', linewidth=1.5, alpha=0.7)
    ax.plot([camera_pos[0, 0], camera_y[0, 0]], 
            [camera_pos[1, 0], camera_y[1, 0]], 
            [camera_pos[2, 0], camera_y[2, 0]], 'g-', linewidth=1.5, alpha=0.7)
    ax.plot([camera_pos[0, 0], camera_z[0, 0]], 
            [camera_pos[1, 0], camera_z[1, 0]], 
            [camera_pos[2, 0], camera_z[2, 0]], 'b-', linewidth=1.5, alpha=0.7)
    
    # Plot line from camera to marker center
    ax.plot([camera_pos[0, 0], 0], 
            [camera_pos[1, 0], 0], 
            [camera_pos[2, 0], 0], 'k--', alpha=0.3, linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Pose Relative to ArUco Marker')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = max(
        abs(camera_pos[0, 0]) + marker_size,
        abs(camera_pos[1, 0]) + marker_size,
        abs(camera_pos[2, 0]) + marker_size
    )
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range/2, max_range])
    
    # Add text info
    distance = np.linalg.norm(tvec)
    info_text = f"Distance to marker: {distance*100:.2f} cm\n"
    info_text += f"Camera position: ({camera_pos[0,0]:.3f}, {camera_pos[1,0]:.3f}, {camera_pos[2,0]:.3f}) m"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Detect ArUco markers and visualize camera pose')
    parser.add_argument('-d', '--dir', type=str, default='cameras/wide_angle_res3/test_images',
                        help='Directory containing test images')
    parser.add_argument('-c', '--calibration', type=str, default='cameras/wide_angle_res3/results/calibration.txt',
                        help='Path to calibration file')
    parser.add_argument('-s', '--marker-size', type=float, default=0.1,
                        help='ArUco marker size in meters (default: 0.1m = 10cm)')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all images with detections, not just first')
    args = parser.parse_args()
    
    # Load calibration
    print(f"📐 Loading calibration from: {args.calibration}")
    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    print(f"   Camera matrix loaded: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
    
    # Initialize ArUco detector (support both old and new API)
    try:
        # New API (OpenCV 4.7+)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
    except AttributeError:
        # Old API
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
    
    # Find test images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(args.dir, ext)))
    
    if not image_paths:
        print(f"❌ No images found in {args.dir}")
        return
    
    print(f"🖼️  Found {len(image_paths)} test images")
    print(f"🎯 Looking for 4x4 ArUco markers (marker size: {args.marker_size}m)")
    print("-" * 60)
    
    first_detection = None
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_name = os.path.basename(img_path)
        
        # Detect markers
        corners, ids, rejected = detect_aruco_markers(img, aruco_dict, parameters)
        
        if ids is not None and len(ids) > 0:
            print(f"✓ {img_name:40s} - Found {len(ids)} marker(s): {ids.ravel().tolist()}")
            
            # Draw detected markers
            img_display = img.copy()
            try:
                # New API
                cv2.aruco.drawDetectedMarkers(img_display, corners, ids)
            except:
                # Old API or fallback
                img_display = cv2.aruco.drawDetectedMarkers(img_display, corners, ids)
            
            # Process first marker
            if first_detection is None or args.show_all:
                marker_corners = corners[0].reshape(-1, 2)
                rvec, tvec = estimate_pose(marker_corners, args.marker_size, camera_matrix, dist_coeffs)
                
                if rvec is not None:
                    # Draw axis
                    img_display = draw_axis(img_display, camera_matrix, dist_coeffs, rvec, tvec, args.marker_size * 0.5)
                    
                    # Save first detection for 3D plot
                    if first_detection is None:
                        first_detection = {
                            'rvec': rvec,
                            'tvec': tvec,
                            'image': img_display,
                            'name': img_name
                        }
                        print(f"   → First detection saved for 3D visualization")
                    
                    # Display image
                    cv2.imshow(f'ArUco Detection - {img_name}', img_display)
                    cv2.waitKey(500 if args.show_all else 0)
        else:
            print(f"⚠️  {img_name:40s} - No markers found")
    
    cv2.destroyAllWindows()
    
    # Plot 3D visualization of first detection
    if first_detection:
        print("\n" + "=" * 60)
        print(f"📊 Generating 3D pose visualization for: {first_detection['name']}")
        print("=" * 60)
        
        fig = plot_camera_pose_3d(first_detection['rvec'], first_detection['tvec'], args.marker_size)
        plt.show()
    else:
        print("\n❌ No ArUco markers detected in any images")


if __name__ == "__main__":
    main()

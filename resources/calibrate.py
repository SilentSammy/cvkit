import cv2
import numpy as np
import glob
import argparse
import os

# — CONFIG — 
CHECKERBOARD = (8, 5)           # number of inner corners per row, column
square_size  = 0.03             # actual square size in meters (or any unit)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Camera calibration using chessboard pattern')
parser.add_argument('-d', '--dir', type=str, default=r'resources\calibration\attempt3',
                    help='Directory containing calibration images (default: resources\\calibration\\attempt3)')
parser.add_argument('-o', '--output', type=str, default=None,
                    help='Output file path for calibration results (default: auto-generate in results folder)')
parser.add_argument('--show', action='store_true',
                    help='Display detected corners (default: False)')
args = parser.parse_args()

image_dir = args.dir
show_corners = args.show
output_file = args.output

# termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
            30,   # max iterations
            1e-6) # epsilon

# storage for object points and image points
obj_points = []  # 3D points in real world
img_points = []  # 2D points in image plane

# prepare one pattern of object points, e.g. (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# — LOAD IMAGES & FIND CORNERS —
images = [img for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'] 
          for img in glob.glob(os.path.join(image_dir, ext))]

print(f"📁 Searching in: {os.path.abspath(image_dir)}")
print(f"🖼️  Found {len(images)} images")

for fname in images:
    try:
        img = cv2.imread(fname)

        # rotate 90
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find checkerboard corners
        ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH 
                                            + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print(f"⚠️  {os.path.basename(fname):40s} - Corners not found")
            continue

        # refine to subpixel accuracy
        corners_refined = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)

        img_points.append(corners_refined)
        obj_points.append(objp)
        print(f"✓  {os.path.basename(fname):40s} - OK")

        # (optional) draw and display:
        if show_corners:
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ok)
            cv2.imshow('Corners', img)
            cv2.waitKey(10)
    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

if show_corners:
    cv2.destroyAllWindows()

# — RUN STANDARD (PINHOLE) CALIBRATION —
N_OK = len(obj_points)
print(f"\n{'='*60}")
if N_OK < 10:
    print(f"❌ Only {N_OK} valid patterns; need 10–20 good shots.")
    exit()

print(f"✓ Calibrating with {N_OK} valid patterns...")

rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
    obj_points,
    img_points,
    gray.shape[::-1],
    None, None,
    criteria=criteria
)

print(f"\n{'='*60}")
print("CALIBRATION COMPLETE")
print(f"{'='*60}")
print(f"RMS reprojection error: {rms:.6f} pixels")
print(f"Image size: {gray.shape[1]} x {gray.shape[0]}")
print(f"\nIntrinsic matrix K:")
print(K)
print(f"\nDistortion coefficients D (k1, k2, p1, p2, k3):")
print(D.ravel())

# — SAVE RESULTS —
if output_file is None:
    # Auto-generate output path based on input directory
    # e.g., cameras/wide_angle_res3/calibration -> cameras/wide_angle_res3/results/calibration.txt
    parent_dir = os.path.dirname(image_dir)
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, 'calibration.txt')

with open(output_file, 'w') as f:
    f.write("# Camera Calibration Results\n")
    f.write(f"# Image directory: {os.path.abspath(image_dir)}\n")
    f.write(f"# Valid patterns: {N_OK}\n")
    f.write(f"# RMS error: {rms:.6f} pixels\n")
    f.write(f"# Image size: {gray.shape[1]} x {gray.shape[0]}\n")
    f.write(f"# Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners\n")
    f.write(f"# Square size: {square_size} meters\n")
    f.write("\n")
    
    f.write("# Camera Matrix (K)\n")
    f.write("K = np.array([\n")
    for i, row in enumerate(K):
        if i < len(K) - 1:
            f.write(f"    {list(row)},\n")
        else:
            f.write(f"    {list(row)}\n")
    f.write("])\n")
    f.write("\n")
    
    f.write("# Distortion Coefficients (k1, k2, p1, p2, k3)\n")
    f.write(f"D = np.array({list(D.ravel())})\n")
    f.write("\n")
    
    f.write("# Focal lengths\n")
    f.write(f"fx = {K[0, 0]:.8f}\n")
    f.write(f"fy = {K[1, 1]:.8f}\n")
    f.write("\n")
    
    f.write("# Principal point\n")
    f.write(f"cx = {K[0, 2]:.8f}\n")
    f.write(f"cy = {K[1, 2]:.8f}\n")
    f.write("\n")
    
    f.write("# Image dimensions\n")
    f.write(f"width = {gray.shape[1]}\n")
    f.write(f"height = {gray.shape[0]}\n")

print(f"\n✓ Results saved to: {os.path.abspath(output_file)}")
print(f"{'='*60}")
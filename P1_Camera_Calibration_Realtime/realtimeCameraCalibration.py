#!/usr/bin/env python3
"""
Realtime camera calibration and undistortion using a webcam.

Controls:
    SPACE : capture checkerboard frame and save JPG
    C     : calibrate using captured frames
    R     : reset captured calibration frames and saved samples
    S     : save calibration parameters and realtime results
    Q/ESC : quit
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# External webcam index.
CAMERA_INDEX = 2

# Checkerboard inner corners.
#
# Physical checkerboard:
# 9 x 12 squares
#
# Therefore:
# 8 x 11 inner corners
CHECKERBOARD = (8, 11)

# Number of calibration samples required/recommended.
MINIMUM_SAMPLES = 15

# Maximum number of samples.
MAXIMUM_SAMPLES = 30

# Calibration output file.
CALIBRATION_FILE = Path("realtime_calibration.npz")

# Directory for realtime results.
REALTIME_OUTPUT_DIR = Path("output") / "realtime"

# Directory for individual calibration samples.
SAMPLES_DIR = REALTIME_OUTPUT_DIR / "samples"


# ============================================================
# OBJECT POINTS
# ============================================================

object_points_template = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32,
)

object_points_template[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1],
].T.reshape(-1, 2)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

REALTIME_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SAMPLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera index {CAMERA_INDEX}"
    )


# ============================================================
# CALIBRATION DATA
# ============================================================

object_points = []
image_points = []

image_size = None

camera_matrix = None
distortion_coefficients = None

calibrated = False

# Keep the latest frame available for saving realtime results.
latest_frame = None
latest_undistorted = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_checkerboard(frame):
    """
    Detect checkerboard corners in a camera frame.

    Returns:
        gray:
            Grayscale frame.

        found:
            True if checkerboard was detected.

        corners:
            Refined checkerboard corner coordinates.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags,
    )

    if found:

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

    return gray, found, corners


def clear_saved_samples():
    """
    Delete previously saved sample JPG files.
    """

    if not SAMPLES_DIR.exists():
        return

    deleted_count = 0

    for sample_file in SAMPLES_DIR.glob("sample_*.jpg"):

        try:
            sample_file.unlink()
            deleted_count += 1

        except OSError as error:
            print(
                f"Warning: could not delete "
                f"{sample_file}: {error}"
            )

    print(
        f"Deleted {deleted_count} previous sample image(s)."
    )


def clear_realtime_results():
    """
    Delete previously saved realtime result images.
    """

    result_files = [
        REALTIME_OUTPUT_DIR / "realtime_original.jpg",
        REALTIME_OUTPUT_DIR / "realtime_undistorted.jpg",
        REALTIME_OUTPUT_DIR / "realtime_comparison.jpg",
    ]

    deleted_count = 0

    for result_file in result_files:

        if result_file.exists():

            try:
                result_file.unlink()
                deleted_count += 1

            except OSError as error:
                print(
                    f"Warning: could not delete "
                    f"{result_file}: {error}"
                )

    if deleted_count:
        print(
            f"Deleted {deleted_count} previous realtime result image(s)."
        )


def save_sample_image(frame, corners, sample_number):
    """
    Save a captured checkerboard frame with detected
    checkerboard corners drawn on the image.
    """

    sample_image = frame.copy()

    cv2.drawChessboardCorners(
        sample_image,
        CHECKERBOARD,
        corners,
        True,
    )

    filename = (
        SAMPLES_DIR
        / f"sample_{sample_number:02d}.jpg"
    )

    success = cv2.imwrite(
        str(filename),
        sample_image,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save sample image: {filename}"
        )

    print(
        f"Saved sample image: {filename}"
    )


def calibrate_camera():
    """
    Perform camera calibration using captured checkerboard frames.
    """

    global camera_matrix
    global distortion_coefficients
    global calibrated

    if len(object_points) < MINIMUM_SAMPLES:

        print(
            f"Need at least {MINIMUM_SAMPLES} samples. "
            f"Currently: {len(object_points)}"
        )

        return False

    print()
    print("Starting camera calibration...")
    print(f"Samples: {len(object_points)}")
    print(f"Image size: {image_size}")

    rms, camera_matrix, distortion_coefficients, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    )

    calibrated = True

    print()
    print("Calibration completed.")
    print()
    print(
        f"RMS calibration error: {rms:.9f} pixels"
    )

    print()
    print("Camera matrix:")
    print(camera_matrix)

    print()
    print("Distortion coefficients:")
    print(distortion_coefficients)

    print()

    return True


def save_calibration():
    """
    Save calibration parameters and realtime result images.
    """

    if not calibrated:

        print(
            "Camera has not been calibrated yet."
        )

        return

    np.savez(
        CALIBRATION_FILE,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        image_width=image_size[0],
        image_height=image_size[1],
    )

    print()
    print(
        f"Calibration saved to: {CALIBRATION_FILE}"
    )

    # --------------------------------------------------------
    # Save current realtime undistortion results
    # --------------------------------------------------------

    if latest_frame is not None:

        original_image = latest_frame.copy()

        undistorted_image = cv2.undistort(
            latest_frame,
            camera_matrix,
            distortion_coefficients,
        )

        comparison_image = np.hstack(
            (
                original_image,
                undistorted_image,
            )
        )

        original_path = (
            REALTIME_OUTPUT_DIR
            / "realtime_original.jpg"
        )

        undistorted_path = (
            REALTIME_OUTPUT_DIR
            / "realtime_undistorted.jpg"
        )

        comparison_path = (
            REALTIME_OUTPUT_DIR
            / "realtime_comparison.jpg"
        )

        cv2.imwrite(
            str(original_path),
            original_image,
        )

        cv2.imwrite(
            str(undistorted_path),
            undistorted_image,
        )

        cv2.imwrite(
            str(comparison_path),
            comparison_image,
        )

        print(
            f"Original image saved to: {original_path}"
        )

        print(
            f"Undistorted image saved to: {undistorted_path}"
        )

        print(
            f"Comparison image saved to: {comparison_path}"
        )

    print()


def draw_status(frame, found):
    """
    Draw realtime status information on the frame.
    """

    status = (
        f"Samples: {len(object_points)}/{MAXIMUM_SAMPLES}"
    )

    cv2.putText(
        frame,
        status,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    if calibrated:
        calibration_status = "CALIBRATED"
    else:
        calibration_status = "NOT CALIBRATED"

    cv2.putText(
        frame,
        calibration_status,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    if found:
        checkerboard_status = "Checkerboard: FOUND"
    else:
        checkerboard_status = "Checkerboard: NOT FOUND"

    cv2.putText(
        frame,
        checkerboard_status,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "SPACE=Capture  C=Calibrate  R=Reset  S=Save  Q=Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


# ============================================================
# MAIN LOOP
# ============================================================

print("=" * 60)
print("REALTIME CAMERA CALIBRATION")
print("=" * 60)
print()

print(f"Camera index       : {CAMERA_INDEX}")
print(f"Checkerboard       : {CHECKERBOARD}")
print(f"Minimum samples    : {MINIMUM_SAMPLES}")
print(f"Maximum samples    : {MAXIMUM_SAMPLES}")
print()

print("Sample images will be saved to:")
print(f"  {SAMPLES_DIR}")
print()

print("Realtime results will be saved to:")
print(f"  {REALTIME_OUTPUT_DIR}")
print()

print("Controls:")
print("  SPACE = capture checkerboard + save JPG")
print("  C     = perform calibration")
print("  R     = reset samples and delete sample JPGs")
print("  S     = save calibration + realtime results")
print("  Q/ESC = quit")
print()

print("Point the checkerboard toward the camera.")
print("Use different positions and angles.")
print()

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Failed to read frame from camera."
            )

            break

        # Keep latest frame.
        latest_frame = frame.copy()

        if image_size is None:

            image_size = (
                frame.shape[1],
                frame.shape[0],
            )

        gray, found, corners = detect_checkerboard(
            frame
        )

        # ----------------------------------------------------
        # Draw checkerboard detection
        # ----------------------------------------------------

        display_frame = frame.copy()

        if found:

            cv2.drawChessboardCorners(
                display_frame,
                CHECKERBOARD,
                corners,
                found,
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        draw_status(
            display_frame,
            found,
        )

        # ----------------------------------------------------
        # Show original and undistorted after calibration
        # ----------------------------------------------------

        if calibrated:

            undistorted = cv2.undistort(
                frame,
                camera_matrix,
                distortion_coefficients,
            )

            latest_undistorted = (
                undistorted.copy()
            )

            original_display = frame.copy()
            undistorted_display = undistorted.copy()

            cv2.putText(
                original_display,
                "ORIGINAL",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                undistorted_display,
                "UNDISTORTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            comparison = np.hstack(
                (
                    original_display,
                    undistorted_display,
                )
            )

            cv2.imshow(
                "Realtime Original vs Undistorted",
                comparison,
            )

            cv2.imshow(
                "Calibration Status",
                display_frame,
            )

        else:

            cv2.imshow(
                "Realtime Camera Calibration",
                display_frame,
            )

        # ----------------------------------------------------
        # Keyboard controls
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Capture checkerboard sample
        # ----------------------------------------------------

        if key == ord(" "):

            if not found:

                print(
                    "Checkerboard not detected. "
                    "Move the checkerboard into view."
                )

            elif len(object_points) >= MAXIMUM_SAMPLES:

                print(
                    "Maximum number of samples reached."
                )

            else:

                object_points.append(
                    object_points_template.copy()
                )

                image_points.append(
                    corners.copy()
                )

                sample_number = len(object_points)

                save_sample_image(
                    frame,
                    corners,
                    sample_number,
                )

                print(
                    f"Captured calibration sample "
                    f"{sample_number}/{MAXIMUM_SAMPLES}"
                )

                if sample_number == MAXIMUM_SAMPLES:

                    print()
                    print(
                        "Maximum samples reached."
                    )

                    print(
                        "Press C to perform calibration."
                    )

                    print()

        # ----------------------------------------------------
        # Perform calibration
        # ----------------------------------------------------

        elif key == ord("c"):

            calibrate_camera()

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        elif key == ord("r"):

            object_points.clear()
            image_points.clear()

            calibrated = False
            camera_matrix = None
            distortion_coefficients = None

            clear_saved_samples()
            clear_realtime_results()

            print()
            print("Calibration samples reset.")
            print(
                "Saved sample images have also been deleted."
            )
            print()

        # ----------------------------------------------------
        # Save calibration and results
        # ----------------------------------------------------

        elif key == ord("s"):

            save_calibration()

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        elif key == ord("q") or key == 27:

            break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Camera released.")
    print("Program finished.")
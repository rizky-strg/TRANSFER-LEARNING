"""Regression tests for real-time calibration and undistortion helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from cameraCalibrationRealtime import (  # noqa: E402
    calibrate_from_frame_sequence,
    undistort_frame,
)


class RealtimeCalibrationTest(unittest.TestCase):
    """Exercise the live-camera calibration flow with saved image frames."""

    def test_calibrate_from_frame_sequence(self) -> None:
        """A short sequence of valid checkerboard frames produces a usable model."""

        frames = [
            cv2.imread(str(path))
            for path in sorted((PROJECT_DIR / "images").glob("*.jpg"))[:10]
        ]
        frames = [frame for frame in frames if frame is not None]

        calibration = calibrate_from_frame_sequence(frames, display=False)

        self.assertEqual((640, 480), calibration.image_size)
        self.assertGreater(calibration.detected_images, 3)
        self.assertEqual((3, 3), calibration.camera_matrix.shape)
        self.assertTrue(calibration.distortion_coefficients.shape[0] >= 5)

    def test_undistort_frame_uses_calibration(self) -> None:
        """The per-frame undistortion helper returns a same-sized image."""

        calibration = calibrate_from_frame_sequence(
            [
                cv2.imread(str(path))
                for path in sorted((PROJECT_DIR / "images").glob("*.jpg"))[:10]
            ],
            display=False,
        )
        sample = cv2.imread(str((PROJECT_DIR / "images" / "image_10.jpg")))

        undistorted = undistort_frame(calibration, sample, display=False)

        self.assertEqual((480, 640, 3), undistorted.shape)
        self.assertGreater(undistorted.max(), 0)

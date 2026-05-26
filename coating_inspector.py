"""
BRP Coating Coverage Inspector

A Python desktop tool for viewing long conductor TIFF images and estimating
coating coverage from Capacity-N coating platform and eye360 module captures.

Author: Arpys Arevalo
Company: AssetCool
Initial version: 2026-05-26

Purpose:
    - View long stitched conductor TIFF images.
    - Pan, zoom, and auto-pan along the conductor length.
    - Detect white coating coverage.
    - Detect dark gray coating coverage.
    - Export coating coverage profiles as CSV.

Notes:
    This is an early engineering tool intended for internal development,
    validation, and collaborative improvement.
"""

import sys
import csv
import numpy as np
import tifffile as tiff
import cv2

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QMessageBox,
)

import pyqtgraph as pg

# Important: image[y, x] display order
pg.setConfigOptions(imageAxisOrder="row-major")


class ConductorImageProcessor:
    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """
        Convert any image type to display-safe uint8.
        Prevents pyqtgraph float display errors.
        """
        if image is None:
            return None

        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)

        image_float = image.astype(np.float32)

        min_val = np.nanmin(image_float)
        max_val = np.nanmax(image_float)

        if max_val - min_val < 1e-6:
            return np.zeros(image.shape, dtype=np.uint8)

        image_float = (image_float - min_val) / (max_val - min_val)
        image_float = image_float * 255.0

        return np.ascontiguousarray(np.clip(image_float, 0, 255).astype(np.uint8))

    @staticmethod
    def to_gray_uint8(image: np.ndarray) -> np.ndarray:
        img8 = ConductorImageProcessor.normalize_to_uint8(image)

        if img8.ndim == 3:
            if img8.shape[2] == 4:
                img8 = img8[:, :, :3]
            return cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

        return img8

    @staticmethod
    def get_hsv_channels(image: np.ndarray):
        """
        Returns HSV channels from RGB/grayscale image.
        H range: 0-179, S range: 0-255, V range: 0-255.
        """
        img8 = ConductorImageProcessor.normalize_to_uint8(image)

        if img8.ndim == 2:
            rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        else:
            rgb = img8[:, :, :3]

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        return h, s, v

    @staticmethod
    def valid_conductor_mask(image: np.ndarray, background_threshold: int = 20) -> np.ndarray:
        """
        Excludes black gaps/background from the coverage calculation.
        """
        gray = ConductorImageProcessor.to_gray_uint8(image)
        return gray > background_threshold

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
        gray = ConductorImageProcessor.to_gray_uint8(image)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )

        return np.ascontiguousarray(clahe.apply(gray).astype(np.uint8))

    @staticmethod
    def apply_sobel_edges(image: np.ndarray) -> np.ndarray:
        gray = ConductorImageProcessor.to_gray_uint8(image)

        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        mag = np.sqrt(sx**2 + sy**2)

        return ConductorImageProcessor.normalize_to_uint8(mag)

    @staticmethod
    def apply_laplacian(image: np.ndarray) -> np.ndarray:
        gray = ConductorImageProcessor.to_gray_uint8(image)

        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap = np.abs(lap)

        return ConductorImageProcessor.normalize_to_uint8(lap)

    @staticmethod
    def apply_unsharp_mask(image: np.ndarray, amount: float = 1.5) -> np.ndarray:
        img8 = ConductorImageProcessor.normalize_to_uint8(image)

        blurred = cv2.GaussianBlur(img8, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(img8, 1.0 + amount, blurred, -amount, 0)

        return np.ascontiguousarray(np.clip(sharpened, 0, 255).astype(np.uint8))

    @staticmethod
    def flatten_background(image: np.ndarray, blur_sigma: float = 25.0) -> np.ndarray:
        gray = ConductorImageProcessor.to_gray_uint8(image)

        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
        flattened = cv2.subtract(gray, background)
        flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)

        return np.ascontiguousarray(flattened.astype(np.uint8))

    @staticmethod
    def detect_dark_impurities(image: np.ndarray, threshold: int = 50) -> np.ndarray:
        gray = ConductorImageProcessor.to_gray_uint8(image)

        output = np.zeros_like(gray, dtype=np.uint8)
        output[gray < threshold] = 255

        return np.ascontiguousarray(output)

    @staticmethod
    def detect_bright_scratches(image: np.ndarray, threshold: int = 200) -> np.ndarray:
        gray = ConductorImageProcessor.to_gray_uint8(image)

        enhanced = ConductorImageProcessor.apply_laplacian(gray)

        output = np.zeros_like(gray, dtype=np.uint8)
        output[enhanced > threshold] = 255

        return np.ascontiguousarray(output)

    @staticmethod
    def detect_white_coating(
        image: np.ndarray,
        brightness_threshold: int = 170,
        saturation_threshold: int = 80,
        background_threshold: int = 20,
    ) -> np.ndarray:
        """
        Detect likely white coating.

        White coating is expected to be:
        - bright
        - relatively low saturation
        - not part of black gaps/background
        """
        _, s, v = ConductorImageProcessor.get_hsv_channels(image)

        valid = ConductorImageProcessor.valid_conductor_mask(
            image,
            background_threshold=background_threshold,
        )

        mask = (v >= brightness_threshold) & (s <= saturation_threshold) & valid

        output = np.zeros_like(v, dtype=np.uint8)
        output[mask] = 255

        return np.ascontiguousarray(output)

    @staticmethod
    def detect_dark_gray_coating(
        image: np.ndarray,
        dark_min_threshold: int = 25,
        dark_max_threshold: int = 140,
        saturation_threshold: int = 100,
        background_threshold: int = 20,
    ) -> np.ndarray:
        """
        Detect likely dark gray coating.

        Dark gray coating is expected to be:
        - darker than bare aluminium/silver conductor
        - neutral / low-to-moderate saturation
        - brighter than black gaps/background
        """
        _, s, v = ConductorImageProcessor.get_hsv_channels(image)

        valid = ConductorImageProcessor.valid_conductor_mask(
            image,
            background_threshold=background_threshold,
        )

        mask = (
            (v >= dark_min_threshold)
            & (v <= dark_max_threshold)
            & (s <= saturation_threshold)
            & valid
        )

        output = np.zeros_like(v, dtype=np.uint8)
        output[mask] = 255

        return np.ascontiguousarray(output)

    @staticmethod
    def calculate_coverage(
        image: np.ndarray,
        coating_mask: np.ndarray,
        background_threshold: int = 20,
    ) -> dict:
        """
        Calculate coating coverage over all valid conductor pixels.
        """
        valid = ConductorImageProcessor.valid_conductor_mask(
            image,
            background_threshold=background_threshold,
        )

        coated = coating_mask > 0

        valid_pixels = int(np.count_nonzero(valid))
        coated_pixels = int(np.count_nonzero(coated & valid))

        if valid_pixels == 0:
            coverage = 0.0
        else:
            coverage = 100.0 * coated_pixels / valid_pixels

        return {
            "valid_pixels": valid_pixels,
            "coated_pixels": coated_pixels,
            "uncoated_pixels": valid_pixels - coated_pixels,
            "coverage_percent": coverage,
        }

    @staticmethod
    def calculate_coverage_for_region(
        image: np.ndarray,
        coating_mask: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        background_threshold: int = 20,
    ) -> dict:
        """
        Calculate coating coverage only for the selected visible region.
        """
        h, w = image.shape[:2]

        x_min = max(0, min(w, int(x_min)))
        x_max = max(0, min(w, int(x_max)))
        y_min = max(0, min(h, int(y_min)))
        y_max = max(0, min(h, int(y_max)))

        if x_max <= x_min or y_max <= y_min:
            return {
                "valid_pixels": 0,
                "coated_pixels": 0,
                "uncoated_pixels": 0,
                "coverage_percent": 0.0,
            }

        image_roi = image[y_min:y_max, x_min:x_max]
        mask_roi = coating_mask[y_min:y_max, x_min:x_max]

        return ConductorImageProcessor.calculate_coverage(
            image_roi,
            mask_roi,
            background_threshold=background_threshold,
        )

    @staticmethod
    def coating_overlay(
        image: np.ndarray,
        coating_mask: np.ndarray,
        overlay_color=(0, 255, 0),
        alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Generate RGB overlay showing detected coating in colour.

        Note: image is treated as RGB for display.
        """
        img8 = ConductorImageProcessor.normalize_to_uint8(image)

        if img8.ndim == 2:
            rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        else:
            rgb = img8[:, :, :3].copy()

        mask = coating_mask > 0

        overlay = rgb.copy()
        overlay[mask] = overlay_color

        blended = cv2.addWeighted(rgb, 1.0 - alpha, overlay, alpha, 0)

        return np.ascontiguousarray(blended.astype(np.uint8))

    @staticmethod
    def coverage_profile_along_x(
        image: np.ndarray,
        coating_mask: np.ndarray,
        bin_width_px: int = 500,
        background_threshold: int = 20,
        preview_scale_x: int = 1,
    ):
        """
        Calculate coating coverage in vertical slices along the conductor length.
        Returns rows for CSV export.
        """
        h, w = image.shape[:2]

        rows = []

        for x0 in range(0, w, bin_width_px):
            x1 = min(w, x0 + bin_width_px)

            stats = ConductorImageProcessor.calculate_coverage_for_region(
                image=image,
                coating_mask=coating_mask,
                x_min=x0,
                x_max=x1,
                y_min=0,
                y_max=h,
                background_threshold=background_threshold,
            )

            rows.append(
                {
                    "preview_x_start": x0,
                    "preview_x_end": x1,
                    "original_x_start": int(x0 * preview_scale_x),
                    "original_x_end": int(x1 * preview_scale_x),
                    "coverage_percent": stats["coverage_percent"],
                    "coated_pixels": stats["coated_pixels"],
                    "valid_pixels": stats["valid_pixels"],
                    "uncoated_pixels": stats["uncoated_pixels"],
                }
            )

        return rows


class BRPCoatingInspector(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BRP Coating Coverage Inspector")
        self.resize(1700, 1000)

        self.raw_image = None
        self.display_image = None
        self.current_file = None

        self.current_coating_mask = None
        self.last_full_coverage = None
        self.last_visible_coverage = None

        self.original_width = 0
        self.original_height = 0

        # X-only preview scaling. Do not downsample Y.
        # Increase this for higher quality if your Mac handles it.
        self.preview_scale_x = 1
        self.max_preview_width = 120000

        # Initial inspection window width in preview pixels.
        self.initial_visible_width = 1200

        self.auto_pan_timer = QTimer()
        self.auto_pan_timer.timeout.connect(self.auto_pan_step)

        self.auto_pan_pixels_per_tick = 4
        self.auto_pan_interval_ms = 50

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()

        controls_1 = QHBoxLayout()
        controls_2 = QHBoxLayout()
        controls_3 = QHBoxLayout()

        self.open_button = QPushButton("Open TIF")
        self.open_button.clicked.connect(self.open_tif)

        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)

        self.overview_button = QPushButton("Overview")
        self.overview_button.clicked.connect(self.overview_view)

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.export_button = QPushButton("Export Current Image")
        self.export_button.clicked.connect(self.export_processed_image)

        self.export_profile_button = QPushButton("Export Coverage CSV")
        self.export_profile_button.clicked.connect(self.export_coverage_profile_csv)

        self.filter_box = QComboBox()
        self.filter_box.addItems(
            [
                "Original",
                "CLAHE Local Contrast",
                "Flatten Background",
                "Sobel Edges",
                "Laplacian Detail",
                "Sharpen",
                "Dark Impurities",
                "Bright Scratches",
                "White Coating Mask",
                "White Coating Overlay",
                "Dark Gray Coating Mask",
                "Dark Gray Coating Overlay",
                "Valid Conductor Area",
            ]
        )
        self.filter_box.currentTextChanged.connect(self.update_processing)

        self.auto_pan_checkbox = QCheckBox("Auto Pan")
        self.auto_pan_checkbox.stateChanged.connect(self.toggle_auto_pan)

        self.speed_label = QLabel("Pan speed:")
        self.speed_spin = QSpinBox()
        self.speed_spin.setMinimum(1)
        self.speed_spin.setMaximum(500)
        self.speed_spin.setValue(self.auto_pan_pixels_per_tick)
        self.speed_spin.setSuffix(" px/tick")
        self.speed_spin.valueChanged.connect(self.set_pan_speed)

        self.visible_width_label = QLabel("View width:")
        self.visible_width_spin = QSpinBox()
        self.visible_width_spin.setMinimum(100)
        self.visible_width_spin.setMaximum(50000)
        self.visible_width_spin.setSingleStep(100)
        self.visible_width_spin.setValue(self.initial_visible_width)
        self.visible_width_spin.setSuffix(" px")
        self.visible_width_spin.valueChanged.connect(self.set_visible_width_from_spin)

        self.jump_label = QLabel("Jump to original X:")
        self.jump_spin = QSpinBox()
        self.jump_spin.setMinimum(0)
        self.jump_spin.setMaximum(999999999)
        self.jump_spin.setSingleStep(1000)
        self.jump_spin.setValue(0)

        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self.jump_to_original_x)

        self.x_scale_label = QLabel("X scale:")
        self.x_scale_value_label = QLabel("1/1")

        # Coating / segmentation controls
        self.brightness_label = QLabel("White brightness:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(0)
        self.brightness_slider.setMaximum(255)
        self.brightness_slider.setValue(170)
        self.brightness_slider.valueChanged.connect(self.update_processing)

        self.saturation_label = QLabel("Max saturation:")
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setMinimum(0)
        self.saturation_slider.setMaximum(255)
        self.saturation_slider.setValue(90)
        self.saturation_slider.valueChanged.connect(self.update_processing)

        self.dark_min_label = QLabel("Dark min:")
        self.dark_min_slider = QSlider(Qt.Horizontal)
        self.dark_min_slider.setMinimum(0)
        self.dark_min_slider.setMaximum(255)
        self.dark_min_slider.setValue(25)
        self.dark_min_slider.valueChanged.connect(self.update_processing)

        self.dark_max_label = QLabel("Dark max:")
        self.dark_max_slider = QSlider(Qt.Horizontal)
        self.dark_max_slider.setMinimum(0)
        self.dark_max_slider.setMaximum(255)
        self.dark_max_slider.setValue(140)
        self.dark_max_slider.valueChanged.connect(self.update_processing)

        self.background_label = QLabel("Background:")
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setMinimum(0)
        self.background_slider.setMaximum(255)
        self.background_slider.setValue(20)
        self.background_slider.valueChanged.connect(self.update_processing)

        self.profile_bin_label = QLabel("CSV bin width:")
        self.profile_bin_spin = QSpinBox()
        self.profile_bin_spin.setMinimum(10)
        self.profile_bin_spin.setMaximum(100000)
        self.profile_bin_spin.setSingleStep(100)
        self.profile_bin_spin.setValue(500)
        self.profile_bin_spin.setSuffix(" px")

        controls_1.addWidget(self.open_button)
        controls_1.addWidget(self.reset_button)
        controls_1.addWidget(self.overview_button)
        controls_1.addWidget(self.zoom_in_button)
        controls_1.addWidget(self.zoom_out_button)
        controls_1.addWidget(QLabel("Mode:"))
        controls_1.addWidget(self.filter_box)

        controls_2.addWidget(self.auto_pan_checkbox)
        controls_2.addWidget(self.speed_label)
        controls_2.addWidget(self.speed_spin)
        controls_2.addWidget(self.visible_width_label)
        controls_2.addWidget(self.visible_width_spin)
        controls_2.addWidget(self.jump_label)
        controls_2.addWidget(self.jump_spin)
        controls_2.addWidget(self.jump_button)
        controls_2.addWidget(self.x_scale_label)
        controls_2.addWidget(self.x_scale_value_label)
        controls_2.addWidget(self.export_button)

        controls_3.addWidget(self.brightness_label)
        controls_3.addWidget(self.brightness_slider)
        controls_3.addWidget(self.saturation_label)
        controls_3.addWidget(self.saturation_slider)
        controls_3.addWidget(self.dark_min_label)
        controls_3.addWidget(self.dark_min_slider)
        controls_3.addWidget(self.dark_max_label)
        controls_3.addWidget(self.dark_max_slider)
        controls_3.addWidget(self.background_label)
        controls_3.addWidget(self.background_slider)
        controls_3.addWidget(self.profile_bin_label)
        controls_3.addWidget(self.profile_bin_spin)
        controls_3.addWidget(self.export_profile_button)

        self.graphics_layout = pg.GraphicsLayoutWidget()

        self.view = self.graphics_layout.addViewBox()
        self.view.setAspectLocked(False)
        self.view.setMouseEnabled(x=True, y=True)
        self.view.invertY(False)

        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)

        self.status_label = QLabel("Open a conductor TIF file to begin.")

        main_layout.addLayout(controls_1)
        main_layout.addLayout(controls_2)
        main_layout.addLayout(controls_3)
        main_layout.addWidget(self.graphics_layout)
        main_layout.addWidget(self.status_label)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def open_tif(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open conductor TIF image",
            "",
            "TIF Images (*.tif *.tiff);;All Files (*)",
        )

        if not path:
            return

        self.current_file = path
        self.status_label.setText(f"Loading: {path}")
        QApplication.processEvents()

        try:
            with tiff.TiffFile(path) as tif:
                page = tif.pages[0]
                image = page.asarray()

                if image.ndim > 3:
                    image = image[0]

                height, width = image.shape[:2]

                self.original_height = height
                self.original_width = width

                if width > self.max_preview_width:
                    self.preview_scale_x = int(np.ceil(width / self.max_preview_width))

                    if image.ndim == 2:
                        image = image[:, :: self.preview_scale_x]
                    else:
                        image = image[:, :: self.preview_scale_x, :]
                else:
                    self.preview_scale_x = 1

                image = np.ascontiguousarray(image)

        except Exception as exc:
            self.status_label.setText(f"Failed to load TIF: {exc}")
            return

        self.raw_image = image
        self.current_coating_mask = None
        self.last_full_coverage = None
        self.last_visible_coverage = None

        self.jump_spin.setMaximum(max(0, self.original_width - 1))
        self.x_scale_value_label.setText(f"1/{self.preview_scale_x}")

        self.update_processing()
        self.reset_view()

        h, w = self.raw_image.shape[:2]

        self.status_label.setText(
            f"Loaded preview: {w} x {h} px | "
            f"Original: {self.original_width} x {self.original_height} px | "
            f"X scale: 1/{self.preview_scale_x} | "
            f"dtype: {self.raw_image.dtype}"
        )

    def update_processing(self):
        if self.raw_image is None:
            return

        selected_filter = self.filter_box.currentText()

        white_brightness = self.brightness_slider.value()
        saturation_max = self.saturation_slider.value()
        dark_min = self.dark_min_slider.value()
        dark_max = self.dark_max_slider.value()
        background_threshold = self.background_slider.value()

        self.current_coating_mask = None
        self.last_full_coverage = None
        self.last_visible_coverage = None

        try:
            if selected_filter == "Original":
                processed = ConductorImageProcessor.normalize_to_uint8(self.raw_image)

            elif selected_filter == "CLAHE Local Contrast":
                processed = ConductorImageProcessor.apply_clahe(self.raw_image)

            elif selected_filter == "Flatten Background":
                processed = ConductorImageProcessor.flatten_background(self.raw_image)

            elif selected_filter == "Sobel Edges":
                processed = ConductorImageProcessor.apply_sobel_edges(self.raw_image)

            elif selected_filter == "Laplacian Detail":
                processed = ConductorImageProcessor.apply_laplacian(self.raw_image)

            elif selected_filter == "Sharpen":
                processed = ConductorImageProcessor.apply_unsharp_mask(self.raw_image)

            elif selected_filter == "Dark Impurities":
                processed = ConductorImageProcessor.detect_dark_impurities(
                    self.raw_image,
                    threshold=dark_max,
                )

            elif selected_filter == "Bright Scratches":
                processed = ConductorImageProcessor.detect_bright_scratches(
                    self.raw_image,
                    threshold=white_brightness,
                )

            elif selected_filter in ["White Coating Mask", "White Coating Overlay"]:
                mask = ConductorImageProcessor.detect_white_coating(
                    self.raw_image,
                    brightness_threshold=white_brightness,
                    saturation_threshold=saturation_max,
                    background_threshold=background_threshold,
                )

                self.current_coating_mask = mask
                self.last_full_coverage = ConductorImageProcessor.calculate_coverage(
                    self.raw_image,
                    mask,
                    background_threshold=background_threshold,
                )

                if selected_filter == "White Coating Overlay":
                    processed = ConductorImageProcessor.coating_overlay(
                        self.raw_image,
                        mask,
                        overlay_color=(0, 255, 0),
                        alpha=0.35,
                    )
                else:
                    processed = mask

            elif selected_filter in ["Dark Gray Coating Mask", "Dark Gray Coating Overlay"]:
                if dark_max < dark_min:
                    dark_max = dark_min

                mask = ConductorImageProcessor.detect_dark_gray_coating(
                    self.raw_image,
                    dark_min_threshold=dark_min,
                    dark_max_threshold=dark_max,
                    saturation_threshold=saturation_max,
                    background_threshold=background_threshold,
                )

                self.current_coating_mask = mask
                self.last_full_coverage = ConductorImageProcessor.calculate_coverage(
                    self.raw_image,
                    mask,
                    background_threshold=background_threshold,
                )

                if selected_filter == "Dark Gray Coating Overlay":
                    processed = ConductorImageProcessor.coating_overlay(
                        self.raw_image,
                        mask,
                        overlay_color=(0, 255, 0),
                        alpha=0.35,
                    )
                else:
                    processed = mask

            elif selected_filter == "Valid Conductor Area":
                valid = ConductorImageProcessor.valid_conductor_mask(
                    self.raw_image,
                    background_threshold=background_threshold,
                )

                processed = np.zeros(valid.shape, dtype=np.uint8)
                processed[valid] = 255

            else:
                processed = ConductorImageProcessor.normalize_to_uint8(self.raw_image)

        except Exception as exc:
            self.status_label.setText(f"Processing failed: {exc}")
            return

        processed = ConductorImageProcessor.normalize_to_uint8(processed)
        processed = np.ascontiguousarray(processed)

        self.display_image = processed

        self.update_visible_coverage()
        self.display_current_image()
        self.update_status_with_view_info()

    def display_current_image(self):
        if self.display_image is None:
            return

        processed = ConductorImageProcessor.normalize_to_uint8(self.display_image)
        processed = np.ascontiguousarray(processed)

        h, w = processed.shape[:2]

        try:
            if processed.ndim == 3:
                if processed.shape[2] == 4:
                    processed = processed[:, :, :3]

                self.image_item.setImage(
                    processed,
                    autoLevels=False,
                    levels=(0, 255),
                )
            else:
                self.image_item.setImage(
                    processed,
                    autoLevels=False,
                    levels=(0, 255),
                )

            self.image_item.setRect(0, 0, w, h)

        except Exception as exc:
            self.status_label.setText(f"Display failed: {exc}")

    def reset_view(self):
        if self.display_image is None:
            return

        h, w = self.display_image.shape[:2]

        visible_width = min(self.visible_width_spin.value(), w)

        start_x = self.find_first_non_black_x()
        end_x = min(w, start_x + visible_width)

        self.view.setRange(
            xRange=(start_x, end_x),
            yRange=(0, h),
            padding=0.02,
        )

        self.update_visible_coverage()
        self.update_status_with_view_info()

    def overview_view(self):
        if self.display_image is None:
            return

        h, w = self.display_image.shape[:2]

        self.view.setRange(
            xRange=(0, w),
            yRange=(0, h),
            padding=0.02,
        )

        self.update_visible_coverage()
        self.update_status_with_view_info()

    def zoom_in(self):
        if self.display_image is None:
            return

        self.view.scaleBy((0.5, 0.8))
        self.update_visible_coverage()
        self.update_status_with_view_info()

    def zoom_out(self):
        if self.display_image is None:
            return

        self.view.scaleBy((2.0, 1.25))
        self.update_visible_coverage()
        self.update_status_with_view_info()

    def set_visible_width_from_spin(self, value):
        if self.display_image is None:
            return

        view_range = self.view.viewRange()
        x_min, _ = view_range[0]
        y_min, y_max = view_range[1]

        h, w = self.display_image.shape[:2]

        new_x_min = max(0, x_min)
        new_x_max = min(w, new_x_min + value)

        if new_x_max - new_x_min < value and new_x_max == w:
            new_x_min = max(0, w - value)

        self.view.setRange(
            xRange=(new_x_min, new_x_max),
            yRange=(y_min, y_max),
            padding=0,
        )

        self.update_visible_coverage()
        self.update_status_with_view_info()

    def jump_to_original_x(self):
        if self.display_image is None:
            return

        original_x = self.jump_spin.value()
        preview_x = original_x / self.preview_scale_x

        h, w = self.display_image.shape[:2]

        visible_width = min(self.visible_width_spin.value(), w)
        half_width = visible_width / 2

        x_min = max(0, preview_x - half_width)
        x_max = min(w, preview_x + half_width)

        if x_max - x_min < visible_width:
            if x_min == 0:
                x_max = min(w, visible_width)
            elif x_max == w:
                x_min = max(0, w - visible_width)

        self.view.setRange(
            xRange=(x_min, x_max),
            yRange=(0, h),
            padding=0.02,
        )

        self.update_visible_coverage()
        self.update_status_with_view_info()

    def find_first_non_black_x(self):
        if self.display_image is None:
            return 0

        image = self.display_image

        if image.ndim == 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        column_strength = gray.mean(axis=0)

        if column_strength.size == 0:
            return 0

        threshold = max(5, float(column_strength.max()) * 0.05)
        useful_columns = np.where(column_strength > threshold)[0]

        if len(useful_columns) == 0:
            return 0

        return int(useful_columns[0])

    def toggle_auto_pan(self):
        if self.auto_pan_checkbox.isChecked():
            self.auto_pan_timer.start(self.auto_pan_interval_ms)
        else:
            self.auto_pan_timer.stop()

    def set_pan_speed(self, value):
        self.auto_pan_pixels_per_tick = value

    def auto_pan_step(self):
        if self.display_image is None:
            return

        view_range = self.view.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]

        h, w = self.display_image.shape[:2]

        dx = self.auto_pan_pixels_per_tick

        new_x_min = x_min + dx
        new_x_max = x_max + dx

        if new_x_max > w:
            new_x_min = 0
            new_x_max = x_max - x_min

        self.view.setRange(
            xRange=(new_x_min, new_x_max),
            yRange=(y_min, y_max),
            padding=0,
        )

        self.update_visible_coverage()
        self.update_status_with_view_info()

    def get_current_view_bounds(self):
        if self.display_image is None:
            return 0, 0, 0, 0

        h, w = self.display_image.shape[:2]

        view_range = self.view.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]

        x_min = max(0, min(w, int(np.floor(x_min))))
        x_max = max(0, min(w, int(np.ceil(x_max))))
        y_min = max(0, min(h, int(np.floor(y_min))))
        y_max = max(0, min(h, int(np.ceil(y_max))))

        return x_min, x_max, y_min, y_max

    def update_visible_coverage(self):
        if self.raw_image is None or self.current_coating_mask is None:
            self.last_visible_coverage = None
            return

        x_min, x_max, y_min, y_max = self.get_current_view_bounds()
        background_threshold = self.background_slider.value()

        self.last_visible_coverage = ConductorImageProcessor.calculate_coverage_for_region(
            image=self.raw_image,
            coating_mask=self.current_coating_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            background_threshold=background_threshold,
        )

    def update_status_with_view_info(self):
        if self.display_image is None:
            return

        h, w = self.display_image.shape[:2]
        selected_filter = self.filter_box.currentText()

        x_min, x_max, y_min, y_max = self.get_current_view_bounds()

        original_x_min = int(x_min * self.preview_scale_x)
        original_x_max = int(x_max * self.preview_scale_x)

        coverage_text = ""

        if self.last_full_coverage is not None:
            coverage_text += (
                f" | Full coverage: "
                f"{self.last_full_coverage['coverage_percent']:.2f}%"
            )

        if self.last_visible_coverage is not None:
            coverage_text += (
                f" | Visible coverage: "
                f"{self.last_visible_coverage['coverage_percent']:.2f}% "
                f"({self.last_visible_coverage['coated_pixels']} / "
                f"{self.last_visible_coverage['valid_pixels']} px)"
            )

        self.status_label.setText(
            f"Display: {w} x {h} px | "
            f"Original: {self.original_width} x {self.original_height} px | "
            f"X scale: 1/{self.preview_scale_x} | "
            f"Viewing original X: {original_x_min} to {original_x_max} | "
            f"Mode: {selected_filter}"
            f"{coverage_text}"
        )

    def export_processed_image(self):
        if self.display_image is None:
            self.status_label.setText("No processed image to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current image",
            "",
            "TIF Images (*.tif);;PNG Images (*.png);;All Files (*)",
        )

        if not path:
            return

        try:
            export_image = ConductorImageProcessor.normalize_to_uint8(self.display_image)

            if path.lower().endswith(".png"):
                # OpenCV expects BGR, but for grayscale it does not matter.
                if export_image.ndim == 3:
                    export_image_bgr = cv2.cvtColor(export_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path, export_image_bgr)
                else:
                    cv2.imwrite(path, export_image)
            else:
                tiff.imwrite(path, export_image)

            self.status_label.setText(f"Exported image: {path}")

        except Exception as exc:
            self.status_label.setText(f"Failed to export image: {exc}")

    def export_coverage_profile_csv(self):
        if self.raw_image is None:
            QMessageBox.warning(self, "No image", "Open a TIF image first.")
            return

        if self.current_coating_mask is None:
            QMessageBox.warning(
                self,
                "No coating mask",
                "Select one of the coating modes first:\n\n"
                "- White Coating Mask\n"
                "- White Coating Overlay\n"
                "- Dark Gray Coating Mask\n"
                "- Dark Gray Coating Overlay",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export coating coverage profile CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )

        if not path:
            return

        bin_width_px = self.profile_bin_spin.value()
        background_threshold = self.background_slider.value()

        try:
            rows = ConductorImageProcessor.coverage_profile_along_x(
                image=self.raw_image,
                coating_mask=self.current_coating_mask,
                bin_width_px=bin_width_px,
                background_threshold=background_threshold,
                preview_scale_x=self.preview_scale_x,
            )

            with open(path, "w", newline="") as csvfile:
                fieldnames = [
                    "preview_x_start",
                    "preview_x_end",
                    "original_x_start",
                    "original_x_end",
                    "coverage_percent",
                    "coated_pixels",
                    "valid_pixels",
                    "uncoated_pixels",
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in rows:
                    writer.writerow(row)

            self.status_label.setText(f"Exported coverage CSV: {path}")

        except Exception as exc:
            self.status_label.setText(f"Failed to export coverage CSV: {exc}")


def main():
    app = QApplication(sys.argv)
    window = BRPCoatingInspector()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
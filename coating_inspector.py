"""
AssetCool Conductor Coating Inspector

A Python desktop tool for viewing long conductor TIFF images and estimating
coating coverage from Capacity-N coating platform and eye360 module captures.

Author: Arpys Arevalo
Company: AssetCool
Current version: 0.2.2-stable
Last updated: 2026-05-27

Purpose:
    This tool supports engineering review and development of coating quality
    workflows for robotic conductor coating platforms. It is designed to help
    engineers inspect long conductor image acquisitions, tune coating detection
    parameters, estimate coating coverage, identify missing white coating, and
    export coverage profiles along the length of an acquisition.

Main capabilities:
    - Open long stitched TIFF / BigTIFF conductor images.
    - Display very wide conductor images using X-only preview downsampling.
    - Preserve full Y-axis strip detail for stacked camera-strip views.
    - Pan, zoom, jump, and auto-pan along the conductor acquisition.
    - Detect white coating, dark gray coating, and exposed gray conductor.
    - Show coating, uncoated, and missing-coverage overlays.
    - Tune detection parameters using sliders and editable numeric fields.
    - Save and load parameter presets as JSON.
    - Select predefined conductor diameters or enter custom diameters.
    - Estimate distance along the conductor using conductor circumference and
      one-strip image height.
    - Convert auto-pan speed to calibrated m/min.
    - Jump to a calibrated distance in m or cm instead of raw X pixels.
    - Auto-calculate one-strip height when loading a new acquisition.
    - Scroll through the full acquisition length with a calibrated slider above the image.
    - Show dual-colour overlays for detected coated and uncoated regions.
    - Export coverage profile CSV files with pixel and distance metadata.

Important calibration assumption:
    Distance along X is estimated using:

        mm_per_pixel = (pi * conductor_diameter_mm) / conductor_strip_height_px

    where conductor_strip_height_px must be the height of ONE unwrapped conductor
    strip, not the total image height if the image contains multiple stacked
    camera strips.

Notes:
    This is an early internal engineering prototype. Detection is threshold-based
    and should be validated against known reference images before being used for
    formal QA decisions.

    When preview_scale_x > 1 (i.e. very wide images are downsampled for display),
    the coverage CSV bins are computed on the downsampled image. The
    position_start_mm / position_end_mm columns in the CSV are scaled back to
    original coordinates, but the spatial resolution of each bin is coarser than
    those distance values imply. A warning is written to the status bar when
    exporting under these conditions.
"""

import sys
import csv
import json
import math
import numpy as np
import tifffile as tiff
import cv2

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QSlider,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QColorDialog,
)

import pyqtgraph as pg


# row-major makes pyqtgraph's ImageItem interpret numpy arrays as (row, col),
# which matches OpenCV / numpy conventions and avoids a transpose on every
# display update.
pg.setConfigOptions(imageAxisOrder="row-major")


# Ordered mapping of well-known conductor types to their outer diameters in mm.
# Insertion order is preserved (Python 3.7+) and drives the combo-box order.
CONDUCTOR_PRESETS_MM = {
    "TS Killdeer": 43.688,
    "AAAC Totara": 28.98,
    "AAAC Araucaria": 37.26,
    "ACSR Rabbit": 10.05,
    "ACSR Hawk": 21.78,
    "AAAC Redwood": 41.04,
    "ACSR Drake": 28.13,
}


# ---------------------------------------------------------------------------
# Image processing utilities
# ---------------------------------------------------------------------------

class ConductorImageProcessor:
    """
    Stateless image processing helpers for conductor coating inspection.

    All methods are static so they can be unit-tested independently of the
    Qt UI and called from scripts or notebooks without instantiation.
    """

    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """Linearly rescale any numeric dtype to uint8 [0, 255]."""
        if image is None:
            return None

        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)

        image_float = image.astype(np.float32)
        min_val = np.nanmin(image_float)
        max_val = np.nanmax(image_float)

        if max_val - min_val < 1e-6:
            return np.zeros(image.shape, dtype=np.uint8)

        image_float = (image_float - min_val) / (max_val - min_val) * 255.0
        return np.ascontiguousarray(np.clip(image_float, 0, 255).astype(np.uint8))

    @staticmethod
    def to_gray_uint8(image: np.ndarray) -> np.ndarray:
        """Convert any image (colour or grayscale, any dtype) to uint8 grayscale."""
        img8 = ConductorImageProcessor.normalize_to_uint8(image)

        if img8.ndim == 3:
            if img8.shape[2] == 4:
                img8 = img8[:, :, :3]
            return cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

        return img8

    @staticmethod
    def get_hsv_channels(image: np.ndarray):
        """
         Return (H, S, V) channel arrays from any input image.

         Grayscale inputs are promoted to RGB before conversion so the full HSV
         pipeline works uniformly regardless of the source image mode.
         """
        img8 = ConductorImageProcessor.normalize_to_uint8(image)

        if img8.ndim == 2:
            rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        else:
            rgb = img8[:, :, :3]

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    @staticmethod
    def valid_conductor_mask(
        image: np.ndarray,
        background_threshold: int = 20,
    ) -> np.ndarray:
        """
        Return a boolean mask that is True for pixels brighter than
        background_threshold.  Black / near-black regions between stacked
        camera strips are excluded by this mask so they don't pollute
        coverage statistics.
        """
        gray = ConductorImageProcessor.to_gray_uint8(image)
        return gray > background_threshold

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: int = 8,
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)."""
        gray = ConductorImageProcessor.to_gray_uint8(image)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )
        return np.ascontiguousarray(clahe.apply(gray).astype(np.uint8))

    @staticmethod
    def apply_sobel_edges(image: np.ndarray) -> np.ndarray:
        """Return a normalised gradient-magnitude image via Sobel filtering."""
        gray = ConductorImageProcessor.to_gray_uint8(image)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        return ConductorImageProcessor.normalize_to_uint8(mag)

    @staticmethod
    def apply_laplacian(image: np.ndarray) -> np.ndarray:
        """Return absolute-value Laplacian of the image, normalised to uint8."""
        gray = ConductorImageProcessor.to_gray_uint8(image)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        return ConductorImageProcessor.normalize_to_uint8(np.abs(lap))

    @staticmethod
    def apply_unsharp_mask(
        image: np.ndarray,
        amount: float = 1.5,
    ) -> np.ndarray:
        """Sharpen by subtracting a Gaussian-blurred version from the original."""
        img8 = ConductorImageProcessor.normalize_to_uint8(image)
        blurred = cv2.GaussianBlur(img8, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(img8, 1.0 + amount, blurred, -amount, 0)
        return np.ascontiguousarray(np.clip(sharpened, 0, 255).astype(np.uint8))

    @staticmethod
    def flatten_background(
        image: np.ndarray,
        blur_sigma: float = 25.0,
    ) -> np.ndarray:
        """
        Remove slowly-varying illumination by subtracting a heavily blurred
        version of the image and re-normalising.
        """
        gray = ConductorImageProcessor.to_gray_uint8(image)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
        flattened = cv2.subtract(gray, background)
        flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)
        return np.ascontiguousarray(flattened.astype(np.uint8))

    @staticmethod
    def detect_dark_impurities(
        image: np.ndarray,
        threshold: int = 50,
    ) -> np.ndarray:
        """Return a binary mask of pixels darker than *threshold*."""
        gray = ConductorImageProcessor.to_gray_uint8(image)
        output = np.zeros_like(gray, dtype=np.uint8)
        output[gray < threshold] = 255
        return np.ascontiguousarray(output)

    @staticmethod
    def detect_bright_scratches(
        image: np.ndarray,
        threshold: int = 200,
    ) -> np.ndarray:
        """
        Return a binary mask of high-frequency bright features (scratches).
        Uses the Laplacian to isolate sharp transitions before thresholding.
        """
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
        Detect white/light-coloured coating using HSV thresholds.

        A pixel is classified as white coating when:
          - V (brightness) >= brightness_threshold
          - S (saturation) <= saturation_threshold  (near-achromatic)
          - pixel belongs to the valid conductor area (not background)
        """
        _, s, v = ConductorImageProcessor.get_hsv_channels(image)
        valid = ConductorImageProcessor.valid_conductor_mask(
            image, background_threshold=background_threshold
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
        Detect dark-gray material using HSV thresholds.

        This method is intentionally reused for two UI modes:
          - "Dark Gray Coating"   — identifies an intentionally dark coating.
          - "Missing Coverage"    — identifies exposed bare conductor, which
                                    appears as dark gray when the white coating
                                    is absent.

        Both modes use the same brightness-range + low-saturation criterion;
        they differ only in how the result is labelled and coloured in the UI.

        A pixel is classified when:
          - dark_min_threshold <= V <= dark_max_threshold
          - S <= saturation_threshold
          - pixel belongs to the valid conductor area (not background)
        """
        _, s, v = ConductorImageProcessor.get_hsv_channels(image)
        valid = ConductorImageProcessor.valid_conductor_mask(
            image, background_threshold=background_threshold
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
        Compute coating coverage statistics for a full image.

        Returns a dict with keys:
            valid_pixels, coated_pixels, uncoated_pixels, coverage_percent
        """
        valid = ConductorImageProcessor.valid_conductor_mask(
            image, background_threshold=background_threshold
        )
        coated = coating_mask > 0
        valid_pixels = int(np.count_nonzero(valid))
        coated_pixels = int(np.count_nonzero(coated & valid))

        coverage = 100.0 * coated_pixels / valid_pixels if valid_pixels > 0 else 0.0

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
        Compute coverage statistics for a rectangular sub-region of an image.

        Coordinates are clamped to valid image bounds before slicing.
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

        return ConductorImageProcessor.calculate_coverage(
            image[y_min:y_max, x_min:x_max],
            coating_mask[y_min:y_max, x_min:x_max],
            background_threshold=background_threshold,
        )

    @staticmethod
    def coating_overlay(
        image: np.ndarray,
        coating_mask: np.ndarray,
        overlay_color=(0, 255, 0),
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Blend a solid-colour overlay onto coated pixels."""
        img8 = ConductorImageProcessor.normalize_to_uint8(image)
        rgb = (
            cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
            if img8.ndim == 2
            else img8[:, :, :3].copy()
        )
        overlay = rgb.copy()
        overlay[coating_mask > 0] = overlay_color
        return np.ascontiguousarray(
            cv2.addWeighted(rgb, 1.0 - alpha, overlay, alpha, 0).astype(np.uint8)
        )

    @staticmethod
    def uncoated_overlay(
        image: np.ndarray,
        coating_mask: np.ndarray,
        background_threshold: int = 20,
        overlay_color=(255, 0, 0),
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Blend a solid-colour overlay onto uncoated (but valid) pixels."""
        img8 = ConductorImageProcessor.normalize_to_uint8(image)
        rgb = (
            cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
            if img8.ndim == 2
            else img8[:, :, :3].copy()
        )
        valid = ConductorImageProcessor.valid_conductor_mask(
            image, background_threshold=background_threshold
        )
        uncoated = valid & (~(coating_mask > 0))
        overlay = rgb.copy()
        overlay[uncoated] = overlay_color
        return np.ascontiguousarray(
            cv2.addWeighted(rgb, 1.0 - alpha, overlay, alpha, 0).astype(np.uint8)
        )

    @staticmethod
    def dual_coated_uncoated_overlay(
        image: np.ndarray,
        coating_mask: np.ndarray,
        background_threshold: int = 20,
        coated_color=(0, 255, 0),
        uncoated_color=(255, 0, 0),
        alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Generate one RGB overlay showing:
            coated pixels   -> coated_color
            uncoated pixels -> uncoated_color

        Uncoated pixels are valid conductor pixels not present in the coating
        mask.  Black/background gaps are excluded by the valid conductor mask.
        """
        img8 = ConductorImageProcessor.normalize_to_uint8(image)
        rgb = (
            cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
            if img8.ndim == 2
            else img8[:, :, :3].copy()
        )
        valid = ConductorImageProcessor.valid_conductor_mask(
            image, background_threshold=background_threshold
        )
        coated = coating_mask > 0
        uncoated = valid & (~coated)
        overlay = rgb.copy()
        overlay[coated] = coated_color
        overlay[uncoated] = uncoated_color
        return np.ascontiguousarray(
            cv2.addWeighted(rgb, 1.0 - alpha, overlay, alpha, 0).astype(np.uint8)
        )

    @staticmethod
    def coverage_profile_along_x(
        image: np.ndarray,
        coating_mask: np.ndarray,
        bin_width_px: int = 500,
        background_threshold: int = 20,
        preview_scale_x: int = 1,
    ):
        """
        Compute per-column-bin coverage statistics along the X axis.

        Parameters
        ----------
        image, coating_mask : np.ndarray
            Both must be in preview (downsampled) pixel space.
        bin_width_px : int
            Width of each coverage bin in preview pixels.
        background_threshold : int
            Passed through to calculate_coverage_for_region.
        preview_scale_x : int
            Downsampling factor applied when loading the image.  Used to
            back-calculate original pixel coordinates for the CSV columns
            original_x_start and original_x_end.

        Note
        ----
        When preview_scale_x > 1 the position_*_mm values in the CSV are
        derived from original-space coordinates, but the actual coverage
        computation runs on the downsampled image.  Bin spatial resolution is
        therefore (bin_width_px * preview_scale_x) original pixels wide.
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


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class AssetCoolCoatingInspector(QMainWindow):
    """
    Main window for the AssetCool Conductor Coating Inspector.

    Instance state
    --------------
    raw_image : np.ndarray or None
        The (possibly X-downsampled) image currently in memory.
    display_image : np.ndarray or None
        The processed / filtered image shown in the viewport.
    current_coating_mask : np.ndarray or None
        Binary mask produced by the active coating-detection mode.
        None when the active mode does not produce a mask (e.g. "Original").
    last_full_coverage : dict or None
        Coverage statistics computed over the entire raw_image.
    last_visible_coverage : dict or None
        Coverage statistics computed over the currently visible viewport region.
    original_width / original_height : int
        Pixel dimensions of the on-disk image before any downsampling.
    preview_scale_x : int
        X downsampling factor (1 = no downsampling).  Always >= 1.
    coated_overlay_color / uncoated_overlay_color : tuple(int, int, int)
        RGB colours used for the dual-overlay display modes.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AssetCool Conductor Coating Inspector")
        self.resize(1750, 1050)

        # --- image state ---
        self.raw_image = None
        self.display_image = None
        self.current_file = None

        self.current_coating_mask = None
        self.last_full_coverage = None
        self.last_visible_coverage = None

        # User-selectable overlay colours (RGB tuples).
        self.coated_overlay_color = (0, 255, 0)
        self.uncoated_overlay_color = (255, 0, 0)

        # --- dimension tracking ---
        self.original_width = 0
        self.original_height = 0

        # X downsampling factor applied when loading a very wide image.
        # preview_scale_x = ceil(original_width / max_preview_width).
        self.preview_scale_x = 1
        self.max_preview_width = 120_000  # pixels; beyond this we downsample X

        # Default pixel width shown in the viewport on first load.
        self.initial_visible_width = 1200

        # --- ruler overlay bookkeeping ---
        # All PlotDataItem / TextItem objects drawn for the X-axis ruler are
        # tracked here so they can be cleared before each redraw.
        self.position_ruler_items = []

        # --- auto-pan ---
        self.auto_pan_timer = QTimer()
        self.auto_pan_timer.timeout.connect(self.auto_pan_step)

        self.auto_pan_speed_m_per_min = 1.0
        self.auto_pan_interval_ms = 50  # tick period in ms

        # --- debounce timers ---
        # Slider changes are debounced so heavy processing only fires after
        # the user pauses (120 ms), not on every incremental drag event.
        self.slider_update_timer = QTimer()
        self.slider_update_timer.setSingleShot(True)
        self.slider_update_timer.timeout.connect(self.update_processing)

        # Ruler overlay is similarly debounced to avoid redrawing on every
        # pan/zoom event while the user is still interacting.
        self.overlay_update_timer = QTimer()
        self.overlay_update_timer.setSingleShot(True)
        self.overlay_update_timer.timeout.connect(self.update_position_overlay)

        self.init_ui()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    def create_parameter_control(
        self,
        title: str,
        slider: QSlider,
        spinbox: QSpinBox,
    ) -> QWidget:
        """
        Build a compact two-row parameter control:
            Row 1: <title label>  <numeric spinbox>
            Row 2: <horizontal slider>

        The slider and spinbox are kept in sync via on_slider_value_changed /
        on_parameter_spinbox_changed; they should already be constructed and
        wired before this helper is called.
        """
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(4)

        title_label = QLabel(title)
        title_label.setMinimumWidth(105)
        spinbox.setFixedWidth(65)

        top_row.addWidget(title_label)
        top_row.addWidget(spinbox)
        top_row.addStretch()

        slider.setMinimumWidth(160)

        layout.addLayout(top_row)
        layout.addWidget(slider)

        widget.setLayout(layout)
        widget.setMinimumWidth(230)
        widget.setMaximumHeight(58)
        return widget

    def init_ui(self):
        """Construct and lay out all widgets."""
        central = QWidget()
        main_layout = QVBoxLayout()

        # --- row 1: file / view controls + mode selector ---
        controls_1 = QHBoxLayout()
        controls_2 = QHBoxLayout()

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

        self.export_button = QPushButton("Export Image")
        self.export_button.clicked.connect(self.export_processed_image)

        self.export_profile_button = QPushButton("Export Coverage CSV")
        self.export_profile_button.clicked.connect(self.export_coverage_profile_csv)

        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.clicked.connect(self.save_settings)

        self.load_settings_button = QPushButton("Load Settings")
        self.load_settings_button.clicked.connect(self.load_settings)

        self.coated_color_button = QPushButton("Coated Colour")
        self.coated_color_button.clicked.connect(self.choose_coated_overlay_color)

        self.uncoated_color_button = QPushButton("Uncoated Colour")
        self.uncoated_color_button.clicked.connect(self.choose_uncoated_overlay_color)

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
                "White Uncoated Overlay",
                "White Coated + Uncoated Overlay",
                "White Missing Coverage Mask",
                "White Missing Coverage Overlay",
                "Dark Gray Coating Mask",
                "Dark Gray Coating Overlay",
                "Dark Gray Uncoated Overlay",
                "Dark Gray Coated + Uncoated Overlay",
                "Valid Conductor Area",
            ]
        )
        self.filter_box.currentTextChanged.connect(self.update_processing)
        self.filter_box.setMinimumWidth(320)

        # --- auto-pan controls ---
        self.auto_pan_checkbox = QCheckBox("Auto Pan")
        self.auto_pan_checkbox.stateChanged.connect(self.toggle_auto_pan)

        self.speed_label = QLabel("Auto-pan speed:")
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setMinimum(0.001)
        self.speed_spin.setMaximum(1000.0)
        self.speed_spin.setDecimals(3)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(self.auto_pan_speed_m_per_min)
        self.speed_spin.setSuffix(" m/min")
        self.speed_spin.valueChanged.connect(self.set_pan_speed)
        self.speed_spin.setMinimumWidth(125)

        self.visible_width_label = QLabel("View width:")
        self.visible_width_spin = QSpinBox()
        self.visible_width_spin.setMinimum(100)
        self.visible_width_spin.setMaximum(50000)
        self.visible_width_spin.setSingleStep(100)
        self.visible_width_spin.setValue(self.initial_visible_width)
        self.visible_width_spin.setSuffix(" px")
        self.visible_width_spin.valueChanged.connect(self.set_visible_width_from_spin)
        self.visible_width_spin.setMinimumWidth(110)

        # --- jump-to-distance controls ---
        self.jump_label = QLabel("Jump distance:")
        self.jump_spin = QDoubleSpinBox()
        self.jump_spin.setMinimum(0.0)
        self.jump_spin.setMaximum(999_999.0)
        self.jump_spin.setDecimals(3)
        self.jump_spin.setSingleStep(0.1)
        self.jump_spin.setValue(0.0)
        self.jump_spin.setMinimumWidth(130)

        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self.jump_to_original_x)

        self.x_scale_label = QLabel("X scale:")
        self.x_scale_value_label = QLabel("1/1")

        # --- detection parameter sliders + paired spinboxes ---
        self.brightness_slider = self.create_parameter_slider(120)
        self.brightness_value_spin = self.create_parameter_spinbox(120)

        self.saturation_slider = self.create_parameter_slider(50)
        self.saturation_value_spin = self.create_parameter_spinbox(50)

        self.dark_min_slider = self.create_parameter_slider(20)
        self.dark_min_value_spin = self.create_parameter_spinbox(20)

        self.dark_max_slider = self.create_parameter_slider(120)
        self.dark_max_value_spin = self.create_parameter_spinbox(120)

        self.background_slider = self.create_parameter_slider(20)
        self.background_value_spin = self.create_parameter_spinbox(20)

        # --- CSV export ---
        self.profile_bin_label = QLabel("CSV bin width:")
        self.profile_bin_spin = QSpinBox()
        self.profile_bin_spin.setMinimum(10)
        self.profile_bin_spin.setMaximum(100_000)
        self.profile_bin_spin.setSingleStep(100)
        self.profile_bin_spin.setValue(500)
        self.profile_bin_spin.setSuffix(" px")
        self.profile_bin_spin.setMinimumWidth(110)

        # --- conductor / calibration controls ---
        self.conductor_combo = QComboBox()
        self.conductor_combo.addItems(list(CONDUCTOR_PRESETS_MM.keys()) + ["Custom"])
        self.conductor_combo.setCurrentText("ACSR Drake")
        self.conductor_combo.currentTextChanged.connect(self.on_conductor_changed)
        self.conductor_combo.setMinimumWidth(180)

        self.conductor_diameter_spin = QDoubleSpinBox()
        self.conductor_diameter_spin.setMinimum(0.01)
        self.conductor_diameter_spin.setMaximum(200.0)
        self.conductor_diameter_spin.setDecimals(3)
        self.conductor_diameter_spin.setSingleStep(0.1)
        self.conductor_diameter_spin.setSuffix(" mm")
        self.conductor_diameter_spin.setValue(CONDUCTOR_PRESETS_MM["ACSR Drake"])
        self.conductor_diameter_spin.valueChanged.connect(self.on_calibration_changed)
        self.conductor_diameter_spin.setMinimumWidth(120)

        self.strip_height_spin = QSpinBox()
        self.strip_height_spin.setMinimum(1)
        self.strip_height_spin.setMaximum(100_000)
        self.strip_height_spin.setSingleStep(1)
        self.strip_height_spin.setValue(700)
        self.strip_height_spin.setSuffix(" px")
        self.strip_height_spin.valueChanged.connect(self.on_calibration_changed)
        self.strip_height_spin.setMinimumWidth(110)

        self.auto_strip_button = QPushButton("Auto Strip Height")
        self.auto_strip_button.clicked.connect(self.auto_estimate_strip_height)

        self.position_unit_combo = QComboBox()
        self.position_unit_combo.addItems(["m", "cm"])
        self.position_unit_combo.setCurrentText("m")
        self.position_unit_combo.currentTextChanged.connect(self.on_position_unit_changed)

        self.show_position_overlay_checkbox = QCheckBox("Show X-Axis Distance Ruler")
        self.show_position_overlay_checkbox.setChecked(True)
        self.show_position_overlay_checkbox.stateChanged.connect(self.on_calibration_changed)

        # --- acquisition position scroll slider ---
        self.length_scroll_label = QLabel("Acquisition position:")
        self.length_scroll_slider = QSlider(Qt.Horizontal)
        self.length_scroll_slider.setMinimum(0)
        self.length_scroll_slider.setMaximum(10000)
        self.length_scroll_slider.setSingleStep(10)
        self.length_scroll_slider.setPageStep(250)
        self.length_scroll_slider.setValue(0)
        self.length_scroll_slider.valueChanged.connect(self.on_length_scroll_changed)

        self.length_scroll_value_label = QLabel("0.00 m / 0.00 m")
        self.length_scroll_value_label.setMinimumWidth(160)

        # --- assemble top control rows ---
        controls_1.addWidget(self.open_button)
        controls_1.addWidget(self.reset_button)
        controls_1.addWidget(self.overview_button)
        controls_1.addWidget(self.zoom_in_button)
        controls_1.addWidget(self.zoom_out_button)
        controls_1.addWidget(QLabel("Mode:"))
        controls_1.addWidget(self.filter_box, stretch=1)

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
        controls_2.addStretch()
        controls_2.addWidget(self.export_button)
        controls_2.addWidget(self.save_settings_button)
        controls_2.addWidget(self.load_settings_button)

        scroll_controls = QHBoxLayout()
        scroll_controls.setContentsMargins(0, 0, 0, 0)
        scroll_controls.setSpacing(8)
        scroll_controls.addWidget(self.length_scroll_label)
        scroll_controls.addWidget(self.length_scroll_slider, stretch=1)
        scroll_controls.addWidget(self.length_scroll_value_label)

        # --- detection parameter group ---
        parameter_group = QGroupBox("Detection Parameters")
        parameter_grid = QGridLayout()
        parameter_grid.setContentsMargins(6, 4, 6, 4)
        parameter_grid.setHorizontalSpacing(10)
        parameter_grid.setVerticalSpacing(2)

        parameter_grid.addWidget(
            self.create_parameter_control(
                "White brightness", self.brightness_slider, self.brightness_value_spin
            ),
            0, 0,
        )
        parameter_grid.addWidget(
            self.create_parameter_control(
                "Max saturation", self.saturation_slider, self.saturation_value_spin
            ),
            0, 1,
        )
        parameter_grid.addWidget(
            self.create_parameter_control(
                "Dark min", self.dark_min_slider, self.dark_min_value_spin
            ),
            0, 2,
        )
        parameter_grid.addWidget(
            self.create_parameter_control(
                "Dark max", self.dark_max_slider, self.dark_max_value_spin
            ),
            0, 3,
        )
        parameter_grid.addWidget(
            self.create_parameter_control(
                "Background", self.background_slider, self.background_value_spin
            ),
            0, 4,
        )

        # --- output group (CSV + colour pickers) ---
        output_group = QWidget()
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(4, 2, 4, 2)
        output_layout.setSpacing(3)

        csv_row = QHBoxLayout()
        csv_row.setContentsMargins(0, 0, 0, 0)
        csv_row.setSpacing(4)
        csv_row.addWidget(self.profile_bin_label)
        csv_row.addWidget(self.profile_bin_spin)

        colour_row = QHBoxLayout()
        colour_row.setContentsMargins(0, 0, 0, 0)
        colour_row.setSpacing(4)
        colour_row.addWidget(self.coated_color_button)
        colour_row.addWidget(self.uncoated_color_button)

        self.video_export_button = QPushButton("Export Synced Videos")
        self.video_export_button.clicked.connect(self.export_synchronized_videos)
        self.video_export_button.setMinimumWidth(170)

        output_layout.addLayout(csv_row)
        output_layout.addLayout(colour_row)
        output_layout.addWidget(self.export_profile_button)
        output_layout.addWidget(self.video_export_button)

        output_group.setLayout(output_layout)
        output_group.setMinimumWidth(320)
        output_group.setMaximumHeight(110)
        parameter_grid.addWidget(output_group, 0, 5)
        parameter_group.setLayout(parameter_grid)

        # --- calibration group ---
        calibration_group = QGroupBox("Conductor Position Calibration")
        calibration_layout = QHBoxLayout()
        calibration_layout.addWidget(QLabel("Conductor:"))
        calibration_layout.addWidget(self.conductor_combo)
        calibration_layout.addWidget(QLabel("Diameter:"))
        calibration_layout.addWidget(self.conductor_diameter_spin)
        calibration_layout.addWidget(QLabel("One strip height:"))
        calibration_layout.addWidget(self.strip_height_spin)
        calibration_layout.addWidget(self.auto_strip_button)
        calibration_layout.addWidget(QLabel("Units:"))
        calibration_layout.addWidget(self.position_unit_combo)
        calibration_layout.addWidget(self.show_position_overlay_checkbox)
        calibration_layout.addStretch()
        calibration_group.setLayout(calibration_layout)

        # --- pyqtgraph viewport ---
        self.graphics_layout = pg.GraphicsLayoutWidget()

        self.view = self.graphics_layout.addViewBox()
        self.view.setAspectLocked(False)
        self.view.setMouseEnabled(x=True, y=True)
        self.view.invertY(False)
        self.view.sigRangeChanged.connect(self.on_view_range_changed)

        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)

        # Semi-transparent text overlay showing calibration summary and current
        # position.  Positioned top-left of the viewport, below the ruler ticks.
        self.position_text_item = pg.TextItem(
            text="",
            anchor=(0, 0),
            color="w",
            fill=pg.mkBrush(0, 0, 0, 180),
            border=pg.mkPen(255, 255, 255, 100),
        )
        self.position_text_item.setZValue(203)
        self.view.addItem(self.position_text_item)

        self.status_label = QLabel("Open a conductor TIF file to begin.")

        # Apply a consistent minimum width to all action buttons so the toolbar
        # rows don't look ragged at different system font sizes.
        for button in [
            self.open_button,
            self.reset_button,
            self.overview_button,
            self.zoom_in_button,
            self.zoom_out_button,
            self.export_button,
            self.save_settings_button,
            self.load_settings_button,
            self.coated_color_button,
            self.uncoated_color_button,
            self.export_profile_button,
            self.auto_strip_button,
        ]:
            button.setMinimumWidth(130)

        self.export_profile_button.setMinimumWidth(170)
        self.auto_strip_button.setMinimumWidth(150)

        # --- assemble main layout ---
        main_layout.addLayout(controls_1)
        main_layout.addLayout(controls_2)
        main_layout.addWidget(parameter_group)
        main_layout.addWidget(calibration_group)
        main_layout.addLayout(scroll_controls)
        main_layout.addWidget(self.graphics_layout, stretch=1)
        main_layout.addWidget(self.status_label)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Slider / spinbox factory helpers
    # ------------------------------------------------------------------

    def create_parameter_slider(self, initial_value: int) -> QSlider:
        """Create a horizontal 0–255 slider and wire up its signals."""
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(255)
        slider.setValue(initial_value)
        self.configure_slider(slider)
        return slider

    def create_parameter_spinbox(self, initial_value: int) -> QSpinBox:
        """Create a 0–255 integer spinbox paired with a parameter slider."""
        spin = QSpinBox()
        spin.setMinimum(0)
        spin.setMaximum(255)
        spin.setSingleStep(1)
        spin.setValue(initial_value)
        spin.setFixedWidth(65)
        spin.valueChanged.connect(self.on_parameter_spinbox_changed)
        return spin

    def configure_slider(self, slider: QSlider):
        """Attach common step settings and signals to a parameter slider."""
        slider.setSingleStep(1)
        slider.setPageStep(5)
        slider.setTracking(True)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.valueChanged.connect(self.on_slider_value_changed)
        slider.sliderReleased.connect(self.on_slider_released)

    # ------------------------------------------------------------------
    # Slider / spinbox synchronisation
    # ------------------------------------------------------------------

    def on_slider_value_changed(self):
        """
        Mirror the slider's new value into its paired spinbox immediately (so
        the number updates while dragging), but defer the expensive processing
        step until the user releases or pauses.
        """
        self.update_parameter_spinboxes_from_sliders()

        # If the user is still holding the slider down, skip processing until
        # sliderReleased fires so we don't thrash the CPU on every pixel.
        sender = self.sender()
        if isinstance(sender, QSlider) and sender.isSliderDown():
            return

        self.schedule_slider_processing()

    def on_slider_released(self):
        """Fire processing immediately when the user lets go of a slider."""
        self.update_parameter_spinboxes_from_sliders()
        self.schedule_slider_processing()

    def on_parameter_spinbox_changed(self):
        """
        When the user edits a spinbox directly, push the new value back to the
        corresponding slider (without triggering slider signals) and schedule
        processing.
        """
        sender = self.sender()

        pairs = [
            (self.brightness_value_spin, self.brightness_slider),
            (self.saturation_value_spin, self.saturation_slider),
            (self.dark_min_value_spin, self.dark_min_slider),
            (self.dark_max_value_spin, self.dark_max_slider),
            (self.background_value_spin, self.background_slider),
        ]

        for spinbox, slider in pairs:
            if sender is spinbox:
                self.sync_slider_without_signal(slider, sender.value())
                break

        self.schedule_slider_processing()

    def sync_slider_without_signal(self, slider: QSlider, value: int):
        """Set a slider value without emitting valueChanged."""
        slider.blockSignals(True)
        slider.setValue(int(value))
        slider.blockSignals(False)

    def sync_spinbox_without_signal(self, spinbox: QSpinBox, value: int):
        """Set a spinbox value without emitting valueChanged."""
        spinbox.blockSignals(True)
        spinbox.setValue(int(value))
        spinbox.blockSignals(False)

    def update_parameter_spinboxes_from_sliders(self):
        """Push all slider values to their paired spinboxes (signal-free)."""
        self.sync_spinbox_without_signal(self.brightness_value_spin, self.brightness_slider.value())
        self.sync_spinbox_without_signal(self.saturation_value_spin, self.saturation_slider.value())
        self.sync_spinbox_without_signal(self.dark_min_value_spin, self.dark_min_slider.value())
        self.sync_spinbox_without_signal(self.dark_max_value_spin, self.dark_max_slider.value())
        self.sync_spinbox_without_signal(self.background_value_spin, self.background_slider.value())

    def schedule_slider_processing(self):
        """Start (or restart) the debounce timer for parameter changes."""
        self.slider_update_timer.start(120)

    # ------------------------------------------------------------------
    # Calibration / distance helpers
    # ------------------------------------------------------------------

    def distance_value_to_mm(self, value: float) -> float:
        """Convert a value in the current display unit (m or cm) to mm."""
        unit = self.position_unit_combo.currentText()
        return float(value) * (10.0 if unit == "cm" else 1000.0)

    def distance_mm_to_value(self, distance_mm: float) -> float:
        """Convert mm to the current display unit (m or cm)."""
        unit = self.position_unit_combo.currentText()
        return float(distance_mm) / (10.0 if unit == "cm" else 1000.0)

    def update_distance_controls(self):
        """
        Reconfigure the jump spinbox suffix, step size, and maximum to match
        the currently selected unit and loaded image length.

        The spinbox's current value is preserved in mm and restored in the new
        unit so switching between m and cm doesn't change the target position.
        """
        calibration = self.get_calibration()
        total_mm = max(0.0, self.original_width * calibration["mm_per_pixel"])

        # Preserve current target in mm so unit conversion doesn't move it.
        old_value_mm = self.distance_value_to_mm(self.jump_spin.value())

        unit = self.position_unit_combo.currentText()
        if unit == "cm":
            self.jump_spin.setSuffix(" cm")
            self.jump_spin.setDecimals(2)
            self.jump_spin.setSingleStep(10.0)
        else:
            self.jump_spin.setSuffix(" m")
            self.jump_spin.setDecimals(2)
            self.jump_spin.setSingleStep(0.1)

        # Use 0.001 as the floor so setMaximum never receives 0, which would
        # clamp the spinbox value to 0 and prevent typing a distance before an
        # image is loaded.
        max_value = max(0.001, self.distance_mm_to_value(total_mm))
        self.jump_spin.setMaximum(max_value)

        preserved_value = min(self.distance_mm_to_value(old_value_mm), max_value)
        self.jump_spin.blockSignals(True)
        self.jump_spin.setValue(preserved_value)
        self.jump_spin.blockSignals(False)

    def on_position_unit_changed(self):
        """Respond to the user switching the display unit (m / cm)."""
        self.update_distance_controls()
        self.on_calibration_changed()

    def on_conductor_changed(self):
        """
        Populate the diameter spinbox from the preset when the user selects a
        named conductor type, then refresh calibration-dependent UI.
        """
        name = self.conductor_combo.currentText()

        if name in CONDUCTOR_PRESETS_MM:
            # Block signals so setting the diameter doesn't trigger
            # on_calibration_changed, which would incorrectly flip the combo
            # back to "Custom".
            self.conductor_diameter_spin.blockSignals(True)
            self.conductor_diameter_spin.setValue(CONDUCTOR_PRESETS_MM[name])
            self.conductor_diameter_spin.blockSignals(False)

        self.update_distance_controls()
        self.update_position_overlay()
        self.update_status_with_view_info()
        self.update_length_scroll_from_view()

    def on_calibration_changed(self):
        """
        Called whenever diameter, strip height, unit, or ruler visibility
        changes.  If the diameter no longer matches the selected preset, the
        combo is switched to "Custom" so the label stays accurate.
        """
        if self.conductor_combo.currentText() != "Custom":
            selected_name = self.conductor_combo.currentText()
            selected_diameter = CONDUCTOR_PRESETS_MM.get(selected_name)

            if selected_diameter is not None:
                if abs(self.conductor_diameter_spin.value() - selected_diameter) > 0.001:
                    self.conductor_combo.blockSignals(True)
                    self.conductor_combo.setCurrentText("Custom")
                    self.conductor_combo.blockSignals(False)

        self.update_distance_controls()
        self.update_position_overlay()
        self.update_status_with_view_info()
        self.update_length_scroll_from_view()

    def on_view_range_changed(self):
        """
        Called whenever the viewport is panned or zoomed.

        The ruler redraw is debounced (50 ms) to avoid hammering pyqtgraph
        during fast drags.  The scroll slider is updated immediately so the
        position label tracks in real time.
        """
        self.overlay_update_timer.start(50)
        self.update_length_scroll_from_view()

    def get_calibration(self) -> dict:
        """
        Return all calibration parameters as a dict.

        The key mm_per_pixel is the distance along the conductor (in mm) that
        corresponds to one pixel in the original (non-downsampled) image:

            mm_per_pixel = π × diameter_mm / strip_height_px
        """
        diameter_mm = float(self.conductor_diameter_spin.value())
        strip_height_px = int(self.strip_height_spin.value())

        if strip_height_px <= 0:
            circumference_mm = 0.0
            mm_per_pixel = 0.0
        else:
            circumference_mm = math.pi * diameter_mm
            mm_per_pixel = circumference_mm / strip_height_px

        return {
            "conductor": self.conductor_combo.currentText(),
            "diameter_mm": diameter_mm,
            "circumference_mm": circumference_mm,
            "strip_height_px": strip_height_px,
            "mm_per_pixel": mm_per_pixel,
            "unit": self.position_unit_combo.currentText(),
            "show_overlay": self.show_position_overlay_checkbox.isChecked(),
        }

    def format_distance(self, distance_mm: float) -> str:
        """Format a distance in mm to the active display unit."""
        unit = self.position_unit_combo.currentText()
        if unit == "cm":
            return f"{distance_mm / 10.0:.2f} cm"
        return f"{distance_mm / 1000.0:.2f} m"

    # ------------------------------------------------------------------
    # X-axis ruler overlay
    # ------------------------------------------------------------------

    def clear_position_ruler_overlay(self):
        """
        Remove all ruler PlotDataItem / TextItem objects from the view.

        This must be called before redrawing the ruler so stale items from the
        previous viewport position are not left behind.
        """
        for item in self.position_ruler_items:
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self.position_ruler_items = []

    def choose_tick_step_mm(self, visible_span_mm: float) -> float:
        """
        Return a 'nice' tick spacing in mm that produces roughly 6 ticks across
        the visible span.  Steps are chosen from a human-readable sequence
        (1, 2, 5, 10, 20, 50, …) so labels are easy to read.
        """
        if visible_span_mm <= 0:
            return 100.0

        raw_step = visible_span_mm / 6

        nice_steps = [
            1, 2, 5,
            10, 20, 50,
            100, 200, 500,
            1000, 2000, 5000,
            10000, 20000, 50000,
            100000,
        ]

        for step in nice_steps:
            if step >= raw_step:
                return float(step)

        return float(nice_steps[-1])

    def update_position_overlay(self):
        """
        Redraw the X-axis distance ruler and calibration summary text overlay.

        The ruler is drawn at the top of the visible viewport.  Tick marks and
        distance labels are placed at 'nice' mm intervals derived from the
        current view range and calibration.  Ticks too close to the viewport
        edges are suppressed to avoid clipping.
        """
        if self.display_image is None:
            return

        self.clear_position_ruler_overlay()

        if not self.show_position_overlay_checkbox.isChecked():
            self.position_text_item.setText("")
            return

        calibration = self.get_calibration()
        mm_per_pixel = calibration["mm_per_pixel"]

        if mm_per_pixel <= 0:
            self.position_text_item.setText("")
            return

        view_range = self.view.viewRange()
        vx_min, vx_max = view_range[0]
        vy_min, vy_max = view_range[1]

        visible_width = vx_max - vx_min
        visible_height = vy_max - vy_min

        if visible_width <= 0 or visible_height <= 0:
            return

        # Convert viewport preview-pixel bounds to original-pixel space, then
        # to physical distance.
        original_x_min = max(0.0, vx_min) * self.preview_scale_x
        original_x_max = max(0.0, vx_max) * self.preview_scale_x

        start_mm = original_x_min * mm_per_pixel
        end_mm = original_x_max * mm_per_pixel
        visible_span_mm = max(0.0, end_mm - start_mm)
        total_mm = self.original_width * mm_per_pixel

        # Ruler baseline is drawn near the very top of the viewport.
        ruler_y = vy_min + visible_height * 0.030
        tick_bottom_y = ruler_y + visible_height * 0.020
        label_y = ruler_y + visible_height * 0.026

        pen = pg.mkPen(255, 230, 0, 230, width=2)

        # Horizontal baseline spanning the full visible width.
        baseline = pg.PlotDataItem([vx_min, vx_max], [ruler_y, ruler_y], pen=pen)
        baseline.setZValue(200)
        self.view.addItem(baseline)
        self.position_ruler_items.append(baseline)

        # Draw tick marks and distance labels at nice intervals.
        tick_step_mm = self.choose_tick_step_mm(visible_span_mm)
        first_tick_mm = math.ceil(start_mm / tick_step_mm) * tick_step_mm
        edge_margin = visible_width * 0.040  # suppress ticks within this margin of the edges

        tick_mm = first_tick_mm
        tick_counter = 0

        while tick_mm <= end_mm and tick_counter < 30:
            preview_x = (tick_mm / mm_per_pixel) / self.preview_scale_x

            if (vx_min + edge_margin) <= preview_x <= (vx_max - edge_margin):
                tick = pg.PlotDataItem(
                    [preview_x, preview_x],
                    [ruler_y, tick_bottom_y],
                    pen=pen,
                )
                tick.setZValue(201)
                self.view.addItem(tick)
                self.position_ruler_items.append(tick)

                label = pg.TextItem(
                    text=self.format_distance(tick_mm),
                    color=(255, 230, 0),
                    anchor=(0.5, 0),
                    fill=pg.mkBrush(0, 0, 0, 170),
                )
                label.setZValue(202)
                label.setPos(preview_x, label_y)
                self.view.addItem(label)
                self.position_ruler_items.append(label)

            tick_mm += tick_step_mm
            tick_counter += 1

        # Calibration / position summary text below the ruler ticks.
        summary_text = (
            f"{calibration['conductor']} | "
            f"Diameter {calibration['diameter_mm']:.3f} mm | "
            f"One-strip height {calibration['strip_height_px']} px | "
            f"{calibration['mm_per_pixel']:.4f} mm/px | "
            f"Visible {self.format_distance(start_mm)} to {self.format_distance(end_mm)} | "
            f"Total {self.format_distance(total_mm)}"
        )

        self.position_text_item.setText(summary_text)
        self.position_text_item.setPos(
            vx_min + visible_width * 0.015,
            ruler_y + visible_height * 0.055,
        )
        self.position_text_item.setZValue(203)

    # ------------------------------------------------------------------
    # Auto strip height estimation
    # ------------------------------------------------------------------

    def auto_estimate_strip_height(self):
        """
        Estimate the height of one camera strip from the loaded image.

        The algorithm looks for horizontal bands of valid (non-background)
        pixels.  The median height of detected bands is used as the estimate.
        This handles images with multiple stacked camera strips separated by
        thin black gaps.

        If no bands are detected (e.g. the image is a single uninterrupted
        strip), the image height divided by 3 is used as a conservative
        fallback.
        """
        if self.raw_image is None:
            self.status_label.setText("Open a TIF image before estimating strip height.")
            return

        try:
            valid = ConductorImageProcessor.valid_conductor_mask(
                self.raw_image,
                background_threshold=self.background_slider.value(),
            )

            # Fraction of valid pixels in each row.
            row_fraction = valid.mean(axis=1)

            # A row is part of a strip if at least 2 % of its pixels are valid.
            row_is_strip = row_fraction > 0.02

            segments = []
            start = None

            for idx, is_strip in enumerate(row_is_strip):
                if is_strip and start is None:
                    start = idx
                elif not is_strip and start is not None:
                    end = idx
                    if end - start > 10:  # ignore noise-level blips
                        segments.append((start, end))
                    start = None

            # Handle a strip that runs to the bottom of the image.
            if start is not None:
                end = len(row_is_strip)
                if end - start > 10:
                    segments.append((start, end))

            if not segments:
                estimated = max(1, self.raw_image.shape[0] // 3)
            else:
                heights = [end - start for start, end in segments]
                estimated = int(round(float(np.median(heights))))

            self.strip_height_spin.setValue(estimated)
            self.status_label.setText(
                f"Estimated one-strip height: {estimated} px "
                f"from {len(segments)} detected strip(s)."
            )
            self.update_position_overlay()

        except Exception as exc:
            self.status_label.setText(f"Failed to estimate strip height: {exc}")

    # ------------------------------------------------------------------
    # Acquisition position scroll slider
    # ------------------------------------------------------------------

    def update_length_scroll_from_view(self):
        """
        Synchronise the acquisition-position slider and distance label with
        the current viewport range.

        The slider value is mapped from 0–10000 to represent the fractional
        position of the left edge of the viewport along the full image width.
        Slider signals are blocked while updating to prevent a feedback loop
        with on_length_scroll_changed.
        """
        if self.display_image is None:
            self.length_scroll_slider.blockSignals(True)
            self.length_scroll_slider.setValue(0)
            self.length_scroll_slider.blockSignals(False)
            self.length_scroll_value_label.setText("0.00 m / 0.00 m")
            return

        calibration = self.get_calibration()
        mm_per_pixel = calibration["mm_per_pixel"]

        h, w = self.display_image.shape[:2]
        x_min, x_max, _, _ = self.get_current_view_bounds()

        visible_width = max(1, x_max - x_min)
        max_start = max(1, w - visible_width)

        # Map x_min to 0–10000 range.
        ratio = max(0.0, min(1.0, float(x_min) / float(max_start)))
        slider_value = int(round(ratio * 10000.0))

        self.length_scroll_slider.blockSignals(True)
        self.length_scroll_slider.setValue(slider_value)
        self.length_scroll_slider.blockSignals(False)

        current_mm = x_min * self.preview_scale_x * mm_per_pixel
        total_mm = self.original_width * mm_per_pixel

        self.length_scroll_value_label.setText(
            f"{self.format_distance(current_mm)} / {self.format_distance(total_mm)}"
        )

    def on_length_scroll_changed(self, value: int):
        """
        Respond to the user dragging the acquisition-position scroll slider.

        The slider value (0–10000) is mapped to a viewport X range that
        preserves the current visible width while repositioning along the
        image.
        """
        if self.display_image is None:
            return

        h, w = self.display_image.shape[:2]
        view_range = self.view.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]

        visible_width = max(1.0, x_max - x_min)
        max_start = max(0.0, w - visible_width)

        ratio = max(0.0, min(1.0, float(value) / 10000.0))
        new_x_min = max_start * ratio
        new_x_max = new_x_min + visible_width

        # Clamp to image bounds.
        if new_x_max > w:
            new_x_max = w
            new_x_min = max(0.0, w - visible_width)

        self.view.setRange(
            xRange=(new_x_min, new_x_max),
            yRange=(y_min, y_max),
            padding=0,
        )

        self.update_visible_coverage()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def get_current_settings(self) -> dict:
        """Serialise all user-adjustable parameters to a dict for JSON export."""
        return {
            "version": 4,
            "mode": self.filter_box.currentText(),
            "white_brightness": self.brightness_slider.value(),
            "max_saturation": self.saturation_slider.value(),
            "dark_min": self.dark_min_slider.value(),
            "dark_max": self.dark_max_slider.value(),
            "background": self.background_slider.value(),
            "view_width": self.visible_width_spin.value(),
            "auto_pan_speed_m_per_min": self.speed_spin.value(),
            "csv_bin_width": self.profile_bin_spin.value(),
            "coated_overlay_color": {
                "r": self.coated_overlay_color[0],
                "g": self.coated_overlay_color[1],
                "b": self.coated_overlay_color[2],
            },
            "uncoated_overlay_color": {
                "r": self.uncoated_overlay_color[0],
                "g": self.uncoated_overlay_color[1],
                "b": self.uncoated_overlay_color[2],
            },
            "position_calibration": {
                "conductor": self.conductor_combo.currentText(),
                "diameter_mm": self.conductor_diameter_spin.value(),
                "strip_height_px": self.strip_height_spin.value(),
                "unit": self.position_unit_combo.currentText(),
                "show_overlay": self.show_position_overlay_checkbox.isChecked(),
            },
        }

    def save_settings(self):
        """Prompt for a file path and write current settings as JSON."""
        settings = self.get_current_settings()

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save coating inspector settings",
            "",
            "JSON Files (*.json);;All Files (*)",
        )

        if not path:
            return

        if not path.lower().endswith(".json"):
            path += ".json"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
            self.status_label.setText(f"Saved settings: {path}")
        except Exception as exc:
            self.status_label.setText(f"Failed to save settings: {exc}")

    def load_settings(self):
        """Prompt for a JSON settings file and apply it."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load coating inspector settings",
            "",
            "JSON Files (*.json);;All Files (*)",
        )

        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            self.apply_settings(settings)
            self.status_label.setText(f"Loaded settings: {path}")
        except Exception as exc:
            self.status_label.setText(f"Failed to load settings: {exc}")

    def apply_settings(self, settings: dict):
        """
        Restore UI state from a settings dict.

        Values missing from older JSON files fall back to the current widget
        values so loading a v1/v2/v3 preset doesn't reset unrelated controls.
        """
        # --- detection thresholds ---
        self.sync_slider_without_signal(
            self.brightness_slider,
            int(settings.get("white_brightness", self.brightness_slider.value())),
        )
        self.sync_slider_without_signal(
            self.saturation_slider,
            int(settings.get("max_saturation", self.saturation_slider.value())),
        )
        self.sync_slider_without_signal(
            self.dark_min_slider,
            int(settings.get("dark_min", self.dark_min_slider.value())),
        )
        self.sync_slider_without_signal(
            self.dark_max_slider,
            int(settings.get("dark_max", self.dark_max_slider.value())),
        )
        self.sync_slider_without_signal(
            self.background_slider,
            int(settings.get("background", self.background_slider.value())),
        )
        self.update_parameter_spinboxes_from_sliders()

        # --- display / pan settings ---
        self.visible_width_spin.setValue(
            int(settings.get("view_width", self.visible_width_spin.value()))
        )
        # "pan_speed" is the v1/v2 key name; "auto_pan_speed_m_per_min" is v3+.
        self.speed_spin.setValue(
            float(
                settings.get(
                    "auto_pan_speed_m_per_min",
                    settings.get("pan_speed", self.speed_spin.value()),
                )
            )
        )
        self.profile_bin_spin.setValue(
            int(settings.get("csv_bin_width", self.profile_bin_spin.value()))
        )

        # --- overlay colours ---
        coated_colour = settings.get("coated_overlay_color")
        if isinstance(coated_colour, dict):
            self.coated_overlay_color = (
                int(coated_colour.get("r", self.coated_overlay_color[0])),
                int(coated_colour.get("g", self.coated_overlay_color[1])),
                int(coated_colour.get("b", self.coated_overlay_color[2])),
            )

        uncoated_colour = settings.get("uncoated_overlay_color")
        if isinstance(uncoated_colour, dict):
            self.uncoated_overlay_color = (
                int(uncoated_colour.get("r", self.uncoated_overlay_color[0])),
                int(uncoated_colour.get("g", self.uncoated_overlay_color[1])),
                int(uncoated_colour.get("b", self.uncoated_overlay_color[2])),
            )

        # --- calibration ---
        calibration = settings.get("position_calibration")
        if isinstance(calibration, dict):
            conductor = calibration.get("conductor", self.conductor_combo.currentText())
            if conductor in CONDUCTOR_PRESETS_MM or conductor == "Custom":
                self.conductor_combo.setCurrentText(conductor)

            self.conductor_diameter_spin.setValue(
                float(calibration.get("diameter_mm", self.conductor_diameter_spin.value()))
            )
            self.strip_height_spin.setValue(
                int(calibration.get("strip_height_px", self.strip_height_spin.value()))
            )

            unit = calibration.get("unit", self.position_unit_combo.currentText())
            if unit in ["m", "cm"]:
                self.position_unit_combo.setCurrentText(unit)

            self.show_position_overlay_checkbox.setChecked(
                bool(calibration.get("show_overlay", self.show_position_overlay_checkbox.isChecked()))
            )

        # --- active mode ---
        mode = settings.get("mode")
        if mode:
            index = self.filter_box.findText(mode)
            if index >= 0:
                self.filter_box.setCurrentIndex(index)

        self.update_distance_controls()
        self.update_processing()
        self.update_position_overlay()

    # ------------------------------------------------------------------
    # Colour picker dialogs
    # ------------------------------------------------------------------

    def choose_coated_overlay_color(self):
        """Open a colour picker and update the coated-region overlay colour."""
        color = QColorDialog.getColor(
            QColor(*self.coated_overlay_color),
            self,
            "Choose coated overlay colour",
        )
        if color.isValid():
            self.coated_overlay_color = (color.red(), color.green(), color.blue())
            self.update_processing()

    def choose_uncoated_overlay_color(self):
        """Open a colour picker and update the uncoated-region overlay colour."""
        color = QColorDialog.getColor(
            QColor(*self.uncoated_overlay_color),
            self,
            "Choose uncoated overlay colour",
        )
        if color.isValid():
            self.uncoated_overlay_color = (color.red(), color.green(), color.blue())
            self.update_processing()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def open_tif(self):
        """
        Prompt for a TIF/BigTIFF file, load the first page, apply X
        downsampling if the image is wider than max_preview_width, and
        trigger the initial display pipeline.

        Note on QApplication.processEvents():
            Called once after updating the status label so the user sees
            "Loading…" before the (potentially slow) tifffile read begins.
            This is intentional for a single-threaded prototype; a production
            version should use a QThread worker to avoid blocking the event loop.
        """
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

                # Guard against corrupt TIFF pages that decode to zero dimensions.
                if height == 0 or width == 0:
                    self.status_label.setText(
                        f"Invalid image: zero dimension ({width} x {height} px). "
                        "Check that the file is not corrupt."
                    )
                    return

                self.original_height = height
                self.original_width = width

                # Downsample X only if the image is wider than the display cap.
                # Y is kept at full resolution to preserve strip detail.
                if width > self.max_preview_width:
                    self.preview_scale_x = int(np.ceil(width / self.max_preview_width))
                    image = (
                        image[:, :: self.preview_scale_x]
                        if image.ndim == 2
                        else image[:, :: self.preview_scale_x, :]
                    )
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

        self.update_distance_controls()
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
        self.update_position_overlay()

    # ------------------------------------------------------------------
    # Image processing pipeline
    # ------------------------------------------------------------------

    def update_processing(self):
        """
        Apply the selected filter / detection mode to raw_image and refresh
        the display.

        For coating-detection modes a binary mask is stored in
        self.current_coating_mask and full-image coverage statistics are
        computed into self.last_full_coverage.

        For the dark-gray modes the dark_min / dark_max sliders must satisfy
        dark_min <= dark_max.  If they are inverted the UI is corrected before
        processing: dark_max is clamped to dark_min AND the dark_max slider /
        spinbox are updated so the displayed value stays consistent.
        """
        if self.raw_image is None:
            return

        selected_filter = self.filter_box.currentText()

        white_brightness = self.brightness_slider.value()
        saturation_max = self.saturation_slider.value()
        background_threshold = self.background_slider.value()

        dark_min = self.dark_min_slider.value()
        dark_max = self.dark_max_slider.value()

        # Clamp dark_max >= dark_min and keep the UI in sync.
        if dark_max < dark_min:
            dark_max = dark_min
            self.sync_slider_without_signal(self.dark_max_slider, dark_max)
            self.sync_spinbox_without_signal(self.dark_max_value_spin, dark_max)

        # Reset analytical properties before pass setup
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
                mask = ConductorImageProcessor.detect_dark_impurities(
                    self.raw_image, threshold=dark_max
                )
                processed = mask
                # FIX: Set metrics hooks so analytical processes don't throw tracebacks
                self.current_coating_mask = mask
                self.last_full_coverage = ConductorImageProcessor.calculate_coverage(
                    self.raw_image, mask, background_threshold=background_threshold
                )

            elif selected_filter == "Bright Scratches":
                mask = ConductorImageProcessor.detect_bright_scratches(
                    self.raw_image, threshold=white_brightness
                )
                processed = mask
                # FIX: Tie analytic metadata tracking down cleanly here too
                self.current_coating_mask = mask
                self.last_full_coverage = ConductorImageProcessor.calculate_coverage(
                    self.raw_image, mask, background_threshold=background_threshold
                )

            elif selected_filter in [
                "White Coating Mask",
                "White Coating Overlay",
                "White Uncoated Overlay",
                "White Coated + Uncoated Overlay",
            ]:
                mask = ConductorImageProcessor.detect_white_coating(
                    self.raw_image,
                    brightness_threshold=white_brightness,
                    saturation_threshold=saturation_max,
                    background_threshold=background_threshold,
                )
                self.current_coating_mask = mask
                self.last_full_coverage = ConductorImageProcessor.calculate_coverage(
                    self.raw_image, mask, background_threshold=background_threshold
                )

                if selected_filter == "White Coating Overlay":
                    processed = ConductorImageProcessor.coating_overlay(
                        self.raw_image, mask,
                        overlay_color=self.coated_overlay_color, alpha=0.35,
                    )
                elif selected_filter == "White Uncoated Overlay":
                    processed = ConductorImageProcessor.uncoated_overlay(
                        self.raw_image, mask,
                        background_threshold=background_threshold,
                        overlay_color=self.uncoated_overlay_color, alpha=0.35,
                    )
                elif selected_filter == "White Coated + Uncoated Overlay":
                    processed = ConductorImageProcessor.dual_coated_uncoated_overlay(
                        self.raw_image, mask,
                        background_threshold=background_threshold,
                        coated_color=self.coated_overlay_color,
                        uncoated_color=self.uncoated_overlay_color, alpha=0.35,
                    )
                else:  # "White Coating Mask"
                    processed = mask

            elif selected_filter in [
                "White Missing Coverage Mask",
                "White Missing Coverage Overlay",
            ]:
                mask = ConductorImageProcessor.detect_dark_gray_coating(
                    self.raw_image,
                    dark_min_threshold=dark_min,
                    dark_max_threshold=dark_max,
                    saturation_threshold=saturation_max,
                    background_threshold=background_threshold,
                )
                self.current_coating_mask = mask
                self.last_full_coverage = ConductorImageProcessor.calculate_coverage(
                    self.raw_image, mask, background_threshold=background_threshold
                )

                if selected_filter == "White Missing Coverage Overlay":
                    processed = ConductorImageProcessor.coating_overlay(
                        self.raw_image, mask,
                        overlay_color=self.uncoated_overlay_color, alpha=0.35,
                    )
                else:  # "White Missing Coverage Mask"
                    processed = mask

            elif selected_filter in [
                "Dark Gray Coating Mask",
                "Dark Gray Coating Overlay",
                "Dark Gray Uncoated Overlay",
                "Dark Gray Coated + Uncoated Overlay",
            ]:
                mask = ConductorImageProcessor.detect_dark_gray_coating(
                    self.raw_image,
                    dark_min_threshold=dark_min,
                    dark_max_threshold=dark_max,
                    saturation_threshold=saturation_max,
                    background_threshold=background_threshold,
                )
                self.current_coating_mask = mask
                self.last_full_coverage = ConductorImageProcessor.calculate_coverage(
                    self.raw_image, mask, background_threshold=background_threshold
                )

                if selected_filter == "Dark Gray Coating Overlay":
                    processed = ConductorImageProcessor.coating_overlay(
                        self.raw_image, mask,
                        overlay_color=self.coated_overlay_color, alpha=0.35,
                    )
                elif selected_filter == "Dark Gray Uncoated Overlay":
                    processed = ConductorImageProcessor.uncoated_overlay(
                        self.raw_image, mask,
                        background_threshold=background_threshold,
                        overlay_color=self.uncoated_overlay_color, alpha=0.35,
                    )
                elif selected_filter == "Dark Gray Coated + Uncoated Overlay":
                    processed = ConductorImageProcessor.dual_coated_uncoated_overlay(
                        self.raw_image, mask,
                        background_threshold=background_threshold,
                        coated_color=self.coated_overlay_color,
                        uncoated_color=self.uncoated_overlay_color, alpha=0.35,
                    )
                else:  # "Dark Gray Coating Mask"
                    processed = mask

            elif selected_filter == "Valid Conductor Area":
                valid = ConductorImageProcessor.valid_conductor_mask(
                    self.raw_image, background_threshold=background_threshold
                )
                processed = np.zeros(valid.shape, dtype=np.uint8)
                processed[valid] = 255

            else:
                processed = ConductorImageProcessor.normalize_to_uint8(self.raw_image)

        except Exception as exc:
            self.status_label.setText(f"Processing failed: {exc}")
            return

        # Optimization: static helpers already guarantee output matrices fit uint8 structure perfectly.
        self.display_image = np.ascontiguousarray(processed)

        self.update_visible_coverage()
        self.display_current_image()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    def display_current_image(self):
        """Push display_image to the pyqtgraph ImageItem."""
        if self.display_image is None:
            return

        # OPTIMIZATION: Avoid running a redundant multi-pass array scanning loop over memory here
        processed = self.display_image
        h, w = processed.shape[:2]

        try:
            # Drop alpha channel if present (pyqtgraph handles RGB or grayscale).
            if processed.ndim == 3 and processed.shape[2] == 4:
                processed = processed[:, :, :3]

            self.image_item.setImage(processed, autoLevels=False, levels=(0, 255))
            self.image_item.setRect(0, 0, w, h)
        except Exception as exc:
            self.status_label.setText(f"Display failed: {exc}")

    # ------------------------------------------------------------------
    # View management
    # ------------------------------------------------------------------

    def reset_view(self):
        """
        Set the viewport to show initial_visible_width pixels starting from
        the first non-black column in the image.
        """
        if self.display_image is None:
            return

        h, w = self.display_image.shape[:2]
        visible_width = min(self.visible_width_spin.value(), w)
        start_x = self.find_first_non_black_x()
        end_x = min(w, start_x + visible_width)

        self.view.setRange(xRange=(start_x, end_x), yRange=(0, h), padding=0.02)

        self.update_visible_coverage()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    def overview_view(self):
        """Zoom out to show the entire image in the viewport."""
        if self.display_image is None:
            return

        h, w = self.display_image.shape[:2]
        self.view.setRange(xRange=(0, w), yRange=(0, h), padding=0.02)

        self.update_visible_coverage()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    def zoom_in(self):
        """Zoom in (halve the visible X range, shrink Y slightly)."""
        if self.display_image is None:
            return

        self.view.scaleBy((0.5, 0.8))
        self.update_visible_coverage()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    def zoom_out(self):
        """Zoom out (double the visible X range, grow Y slightly)."""
        if self.display_image is None:
            return

        self.view.scaleBy((2.0, 1.25))
        self.update_visible_coverage()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    def set_visible_width_from_spin(self, value: int):
        """Resize the viewport to exactly *value* preview pixels wide."""
        if self.display_image is None:
            return

        view_range = self.view.viewRange()
        x_min, _ = view_range[0]
        y_min, y_max = view_range[1]
        h, w = self.display_image.shape[:2]

        new_x_min = max(0, x_min)
        new_x_max = min(w, new_x_min + value)

        # If clamped at the right edge, pull x_min left to preserve the width.
        if new_x_max - new_x_min < value and new_x_max == w:
            new_x_min = max(0, w - value)

        self.view.setRange(xRange=(new_x_min, new_x_max), yRange=(y_min, y_max), padding=0)

        self.update_visible_coverage()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    def jump_to_original_x(self):
        """
        Centre the viewport on the user-entered distance value by converting
        it to a preview-pixel X coordinate.
        """
        if self.display_image is None:
            return

        calibration = self.get_calibration()
        mm_per_original_pixel = calibration["mm_per_pixel"]

        if mm_per_original_pixel <= 0:
            self.status_label.setText("Cannot jump: calibration scale is zero.")
            return

        target_distance_mm = self.distance_value_to_mm(self.jump_spin.value())
        original_x = target_distance_mm / mm_per_original_pixel
        preview_x = original_x / max(1, self.preview_scale_x)

        h, w = self.display_image.shape[:2]
        visible_width = min(self.visible_width_spin.value(), w)
        half_width = visible_width / 2

        x_min = max(0, preview_x - half_width)
        x_max = min(w, preview_x + half_width)

        # Maintain visible_width even when close to the image edges.
        if x_max - x_min < visible_width:
            if x_min == 0:
                x_max = min(w, visible_width)
            elif x_max == w:
                x_min = max(0, w - visible_width)

        self.view.setRange(xRange=(x_min, x_max), yRange=(0, h), padding=0.02)

        self.update_visible_coverage()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    def find_first_non_black_x(self) -> int:
        """
        Return the X column index of the first non-black content in
        display_image.  Used by reset_view to skip any leading black fill.
        """
        if self.display_image is None:
            return 0

        image = self.display_image
        gray = (
            cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            if image.ndim == 3
            else image
        )

        column_strength = gray.mean(axis=0)
        if column_strength.size == 0:
            return 0

        threshold = max(5, float(column_strength.max()) * 0.05)
        useful_columns = np.where(column_strength > threshold)[0]

        return int(useful_columns[0]) if len(useful_columns) > 0 else 0

    # ------------------------------------------------------------------
    # Auto-pan
    # ------------------------------------------------------------------

    def toggle_auto_pan(self):
        """Start or stop the auto-pan timer based on the checkbox state."""
        if self.auto_pan_checkbox.isChecked():
            self.auto_pan_timer.start(self.auto_pan_interval_ms)
        else:
            self.auto_pan_timer.stop()

    def set_pan_speed(self, value: float):
        """Update the auto-pan speed in m/min."""
        self.auto_pan_speed_m_per_min = float(value)

    def auto_pan_step(self):
        """
        Advance the viewport by one tick's worth of distance based on the
        calibrated pan speed.  Wraps back to X=0 when the right edge is reached.

        Step size in preview pixels:
            mm_per_tick = speed_m_per_min * 1000 * interval_ms / 60000
            original_pixels_per_tick = mm_per_tick / mm_per_original_pixel
            preview_pixels_per_tick = original_pixels_per_tick / preview_scale_x
        """
        if self.display_image is None:
            return

        calibration = self.get_calibration()
        mm_per_original_pixel = calibration["mm_per_pixel"]

        if mm_per_original_pixel <= 0:
            return

        view_range = self.view.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]
        h, w = self.display_image.shape[:2]

        mm_per_tick = (
            self.auto_pan_speed_m_per_min
            * 1000.0
            * self.auto_pan_interval_ms
            / 60000.0
        )
        dx = (mm_per_tick / mm_per_original_pixel) / max(1, self.preview_scale_x)

        new_x_min = x_min + dx
        new_x_max = x_max + dx

        # Wrap to the beginning when the right edge is reached.
        if new_x_max > w:
            new_x_min = 0.0
            new_x_max = x_max - x_min

        self.view.setRange(xRange=(new_x_min, new_x_max), yRange=(y_min, y_max), padding=0)

        self.update_visible_coverage()
        self.update_status_with_view_info()
        self.update_position_overlay()
        self.update_length_scroll_from_view()

    # ------------------------------------------------------------------
    # Coverage statistics
    # ------------------------------------------------------------------

    def get_current_view_bounds(self) -> tuple:
        """
        Return (x_min, x_max, y_min, y_max) in integer preview-pixel
        coordinates, clamped to the image bounds.
        """
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
        """
        Recompute coverage statistics for the currently visible viewport
        region and store the result in self.last_visible_coverage.
        """
        if self.raw_image is None or self.current_coating_mask is None:
            self.last_visible_coverage = None
            return

        x_min, x_max, y_min, y_max = self.get_current_view_bounds()

        self.last_visible_coverage = ConductorImageProcessor.calculate_coverage_for_region(
            image=self.raw_image,
            coating_mask=self.current_coating_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            background_threshold=self.background_slider.value(),
        )

    def update_status_with_view_info(self):
        """Update the status bar with image dimensions, view position, and coverage."""
        if self.display_image is None:
            return

        h, w = self.display_image.shape[:2]
        selected_filter = self.filter_box.currentText()
        x_min, x_max, _, _ = self.get_current_view_bounds()

        original_x_min = int(x_min * self.preview_scale_x)
        original_x_max = int(x_max * self.preview_scale_x)

        is_missing_mode = selected_filter in [
            "White Missing Coverage Mask",
            "White Missing Coverage Overlay",
        ]

        full_label = "Full missing coverage" if is_missing_mode else "Full coverage"
        visible_label = "Visible missing coverage" if is_missing_mode else "Visible coverage"

        calibration = self.get_calibration()
        mm_per_pixel = calibration["mm_per_pixel"]
        start_mm = original_x_min * mm_per_pixel
        end_mm = original_x_max * mm_per_pixel

        coverage_text = ""

        if self.last_full_coverage is not None:
            coverage_text += (
                f" | {full_label}: {self.last_full_coverage['coverage_percent']:.2f}%"
            )

        if self.last_visible_coverage is not None:
            coverage_text += (
                f" | {visible_label}: "
                f"{self.last_visible_coverage['coverage_percent']:.2f}% "
                f"({self.last_visible_coverage['coated_pixels']} / "
                f"{self.last_visible_coverage['valid_pixels']} px)"
            )

        self.status_label.setText(
            f"Display: {w} x {h} px | "
            f"Original: {self.original_width} x {self.original_height} px | "
            f"X scale: 1/{self.preview_scale_x} | "
            f"Viewing original X: {original_x_min} to {original_x_max} | "
            f"Position: {self.format_distance(start_mm)} to {self.format_distance(end_mm)} | "
            f"Mode: {selected_filter}"
            f"{coverage_text}"
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_processed_image(self):
        """Export the current display_image to TIF or PNG."""
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
                export_bgr = (
                    cv2.cvtColor(export_image, cv2.COLOR_RGB2BGR)
                    if export_image.ndim == 3
                    else export_image
                )
                cv2.imwrite(path, export_bgr)
            else:
                tiff.imwrite(path, export_image)

            self.status_label.setText(f"Exported image: {path}")

        except Exception as exc:
            self.status_label.setText(f"Failed to export image: {exc}")

    def export_coverage_profile_csv(self):
        """
        Export a per-bin coverage profile CSV for the full acquisition.

        The CSV bins are computed on the preview (downsampled) image.  When
        preview_scale_x > 1 each bin spans (bin_width_px × preview_scale_x)
        original pixels, which is noted in the status bar after export.

        Requires an active coating-detection mode to be selected first.
        """
        if self.raw_image is None:
            QMessageBox.warning(self, "No image", "Open a TIF image first.")
            return

        if self.current_coating_mask is None:
            QMessageBox.warning(
                self,
                "No coating mask",
                "Select one of the coating or missing coverage modes first:\n\n"
                "- White Coating Mask\n"
                "- White Coating Overlay\n"
                "- White Uncoated Overlay\n"
                "- White Coated + Uncoated Overlay\n"
                "- White Missing Coverage Mask\n"
                "- White Missing Coverage Overlay\n"
                "- Dark Gray Coating Mask\n"
                "- Dark Gray Coating Overlay\n"
                "- Dark Gray Uncoated Overlay\n"
                "- Dark Gray Coated + Uncoated Overlay\n"
                "- Dark Impurities\n"
                "- Bright Scratches",
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

            calibration = self.get_calibration()
            mm_per_pixel = calibration["mm_per_pixel"]

            for row in rows:
                row["position_start_mm"] = row["original_x_start"] * mm_per_pixel
                row["position_end_mm"] = row["original_x_end"] * mm_per_pixel
                row["conductor"] = calibration["conductor"]
                row["diameter_mm"] = calibration["diameter_mm"]
                row["strip_height_px"] = calibration["strip_height_px"]
                row["mm_per_pixel"] = calibration["mm_per_pixel"]

            with open(path, "w", newline="") as csvfile:
                fieldnames = [
                    "conductor",
                    "diameter_mm",
                    "strip_height_px",
                    "mm_per_pixel",
                    "preview_x_start",
                    "preview_x_end",
                    "original_x_start",
                    "original_x_end",
                    "position_start_mm",
                    "position_end_mm",
                    "coverage_percent",
                    "coated_pixels",
                    "valid_pixels",
                    "uncoated_pixels",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in rows:
                    writer.writerow(row)

            # Warn when the downsampled bin resolution differs from the distance
            # precision implied by the position_*_mm columns.
            if self.preview_scale_x > 1:
                effective_bin_original_px = bin_width_px * self.preview_scale_x
                self.status_label.setText(
                    f"Exported coverage CSV: {path} | "
                    f"Note: X scale 1/{self.preview_scale_x} — each bin covers "
                    f"{effective_bin_original_px} original pixels; "
                    f"position_mm values are back-calculated from original coordinates."
                )
            else:
                self.status_label.setText(f"Exported coverage CSV: {path}")

        except Exception as exc:
            self.status_label.setText(f"Failed to export coverage CSV: {exc}")

    def export_synchronized_videos(self):
        """
        Export two synchronized frame-by-frame MP4 videos for the full length
        of the loaded acquisition asset:
          - A "clean" video tracking the processed filter mode.
          - A "masked" video with the user-selected overlay colours tracking masks.
        """
        if self.display_image is None:
            self.status_label.setText("No active asset image loaded to export video.")
            return

        calibration = self.get_calibration()
        mm_per_original_pixel = calibration["mm_per_pixel"]

        if mm_per_original_pixel <= 0:
            self.status_label.setText(
                "Export aborted: Calibration scale factor is invalid."
            )
            return

        fps = 30
        mm_per_second = (self.auto_pan_speed_m_per_min * 1000.0) / 60.0
        mm_per_frame = mm_per_second / fps
        dx_px = float(
            (mm_per_frame / mm_per_original_pixel) / max(1, self.preview_scale_x)
        )

        # CRITICAL FIX: Explicitly extract and force standard Python int types
        img_h, img_w = self.display_image.shape[:2]
        h = int(img_h)
        w = int(img_w)
        view_w = int(min(self.visible_width_spin.value(), w))

        base_path, _ = QFileDialog.getSaveFileName(
            self, "Save Synchronized Performance Videos", "", "AVI Video (*.avi)"
        )
        if not base_path:
            return

        from pathlib import Path

        pure_path = Path(base_path)
        base_no_ext = pure_path.with_suffix("")

        path_clean = str(base_no_ext) + "_clean.avi"
        path_masked = str(base_no_ext) + "_masked.avi"

        self.status_label.setText(
            "Baking synced inspection streams... GUI may lock up temporarily."
        )
        QApplication.processEvents()

        # CRITICAL FIX: Ensure width and height arguments are clean primitive Python ints
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer_clean = cv2.VideoWriter(path_clean, fourcc, fps, (view_w, h))
        writer_masked = cv2.VideoWriter(path_masked, fourcc, fps, (view_w, h))

        selected_filter = self.filter_box.currentText()
        white_brightness = self.brightness_slider.value()
        saturation_max = self.saturation_slider.value()
        background_threshold = self.background_slider.value()
        dark_min = self.dark_min_slider.value()
        dark_max = self.dark_max_slider.value()

        if selected_filter in [
            "White Coating Mask",
            "White Coating Overlay",
            "White Uncoated Overlay",
            "White Coated + Uncoated Overlay",
        ]:
            mask = ConductorImageProcessor.detect_white_coating(
                self.raw_image, white_brightness, saturation_max, background_threshold
            )
            c_color, uc_color = self.coated_overlay_color, self.uncoated_overlay_color
        elif selected_filter in [
            "White Missing Coverage Mask",
            "White Missing Coverage Overlay",
        ]:
            mask = ConductorImageProcessor.detect_dark_gray_coating(
                self.raw_image, dark_min, dark_max, saturation_max, background_threshold
            )
            c_color, uc_color = self.uncoated_overlay_color, self.uncoated_overlay_color
        elif selected_filter in [
            "Dark Gray Coating Mask",
            "Dark Gray Coating Overlay",
            "Dark Gray Uncoated Overlay",
            "Dark Gray Coated + Uncoated Overlay",
        ]:
            mask = ConductorImageProcessor.detect_dark_gray_coating(
                self.raw_image, dark_min, dark_max, saturation_max, background_threshold
            )
            c_color, uc_color = self.coated_overlay_color, self.uncoated_overlay_color
        else:
            mask = ConductorImageProcessor.detect_bright_scratches(
                self.raw_image, white_brightness
            )
            c_color, uc_color = self.coated_overlay_color, self.uncoated_overlay_color

        clean_view_uint8 = ConductorImageProcessor.normalize_to_uint8(
            self.display_image
        )
        # if clean_view_uint8.ndim == 2:
        #     full_clean_view = cv2.cvtColor(clean_view_uint8, cv2.COLOR_GRAY2BGR)
        # else:
        #     full_clean_view = cv2.cvtColor(
        #         clean_view_uint8[:, :, :3], cv2.COLOR_RGB2BGR
        #     )
        base_gray = ConductorImageProcessor.to_gray_uint8(self.raw_image)
        full_clean_view = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

        overlay_rgb = ConductorImageProcessor.coating_overlay(
            self.raw_image,
            mask,
            overlay_color=(0, 255, 0),  # Strictly pure green
            alpha=0.35,
        )
        full_masked_view = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        # CRITICAL FIX: Forcing initialization to float to allow precise mathematical steps
        current_x = float(self.find_first_non_black_x())

        # SAFETY GUARD: If the first non-black X combined with view_w overflows the bounds,
        # fallback to 0.0 to ensure the loop runs at least a single frame pass.
        if current_x + view_w > w:
            current_x = 0.0

        # 1. Calculate approximate total frames beforehand so we can map the progress percentage
        total_frames = max(1, int(math.ceil((w - view_w - current_x) / dx_px))) + 1

        print(f"\n--- Starting Video Export ---")
        print(f"Target Video Resolution: {view_w}x{h} @ {fps}fps")
        print(f"Processing approximately {total_frames} frames...")

        frame_count = 0
        while current_x + view_w <= w:
            x_start = int(current_x)
            x_end = int(current_x + view_w)

            frame_clean = full_clean_view[:, x_start:x_end]
            frame_masked = full_masked_view[:, x_start:x_end]

            writer_clean.write(frame_clean)
            writer_masked.write(frame_masked)

            current_x += dx_px
            frame_count += 1
            if frame_count > 2000:
                break

            # 2. Print progress to terminal (using \r so it updates on the same line)
            percent = min(100, int((frame_count / total_frames) * 100))
            sys.stdout.write(
                f"\rExport Progress: [{percent}%] rendered {frame_count}/{total_frames} frames"
            )
            sys.stdout.flush()

        writer_clean.release()
        writer_masked.release()

        # 3. Clean wrap-up in the terminal
        print(f"\nExport complete. Files released.\n-----------------------------\n")

        if frame_count == 0:
            self.status_label.setText(
                "Export failed: Layout limits prevented frames from rendering."
            )
        else:
            self.status_label.setText(
                f"Export Success! Wrote {frame_count} frames.\n1. {path_clean}\n2. {path_masked}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    window = AssetCoolCoatingInspector()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
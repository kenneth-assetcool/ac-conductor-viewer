import sys
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
)

import pyqtgraph as pg

# Important: use normal image axis order: image[y, x]
pg.setConfigOptions(imageAxisOrder="row-major")


class ConductorImageProcessor:
    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """
        Convert any image type to display-safe uint8.
        This prevents pyqtgraph float display errors.
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


class BRPConductorViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BRP Conductor TIF Inspection Tool")
        self.resize(1600, 950)

        self.raw_image = None
        self.display_image = None
        self.current_file = None

        self.original_width = 0
        self.original_height = 0

        # Downsample X only. Do not downsample Y.
        self.preview_scale_x = 1

        # Lower value = easier for GUI to handle.
        # Your 456496 px wide image becomes about 28531 px wide at this setting.
        self.max_preview_width = 120000

        # Initial visible window in preview pixels.
        self.initial_visible_width = 2200

        self.auto_pan_timer = QTimer()
        self.auto_pan_timer.timeout.connect(self.auto_pan_step)

        self.auto_pan_pixels_per_tick = 8
        self.auto_pan_interval_ms = 50

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()
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

        self.export_button = QPushButton("Export Processed Image")
        self.export_button.clicked.connect(self.export_processed_image)

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
            ]
        )
        self.filter_box.currentTextChanged.connect(self.update_processing)

        self.threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(80)
        self.threshold_slider.valueChanged.connect(self.update_processing)

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

        controls_1.addWidget(self.open_button)
        controls_1.addWidget(self.reset_button)
        controls_1.addWidget(self.overview_button)
        controls_1.addWidget(self.zoom_in_button)
        controls_1.addWidget(self.zoom_out_button)
        controls_1.addWidget(QLabel("Filter:"))
        controls_1.addWidget(self.filter_box)
        controls_1.addWidget(self.threshold_label)
        controls_1.addWidget(self.threshold_slider)

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

                    # X-only downsample.
                    if image.ndim == 2:
                        image = image[:, :: self.preview_scale_x]
                    else:
                        image = image[:, :: self.preview_scale_x, :]
                else:
                    self.preview_scale_x = 1

                # Ensure safe memory layout.
                image = np.ascontiguousarray(image)

        except Exception as exc:
            self.status_label.setText(f"Failed to load TIF: {exc}")
            return

        self.raw_image = image

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
        threshold = self.threshold_slider.value()

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
                    threshold=threshold,
                )

            elif selected_filter == "Bright Scratches":
                processed = ConductorImageProcessor.detect_bright_scratches(
                    self.raw_image,
                    threshold=threshold,
                )

            else:
                processed = ConductorImageProcessor.normalize_to_uint8(self.raw_image)

        except Exception as exc:
            self.status_label.setText(f"Processing failed: {exc}")
            return

        # Critical safety step:
        # pyqtgraph must receive uint8 or explicit levels for float images.
        processed = ConductorImageProcessor.normalize_to_uint8(processed)
        processed = np.ascontiguousarray(processed)

        self.display_image = processed

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
            return

        self.update_status_with_view_info()

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

        self.update_status_with_view_info()

    def zoom_in(self):
        if self.display_image is None:
            return

        self.view.scaleBy((0.5, 0.8))
        self.update_status_with_view_info()

    def zoom_out(self):
        if self.display_image is None:
            return

        self.view.scaleBy((2.0, 1.25))
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

        self.update_status_with_view_info()

    def update_status_with_view_info(self):
        if self.display_image is None:
            return

        h, w = self.display_image.shape[:2]
        selected_filter = self.filter_box.currentText()

        view_range = self.view.viewRange()
        x_min, x_max = view_range[0]

        original_x_min = int(max(0, x_min) * self.preview_scale_x)
        original_x_max = int(min(w, x_max) * self.preview_scale_x)

        self.status_label.setText(
            f"Display: {w} x {h} px | "
            f"Original: {self.original_width} x {self.original_height} px | "
            f"X scale: 1/{self.preview_scale_x} | "
            f"Viewing original X: {original_x_min} to {original_x_max} | "
            f"Filter: {selected_filter}"
        )

    def export_processed_image(self):
        if self.display_image is None:
            self.status_label.setText("No processed image to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export processed image",
            "",
            "TIF Images (*.tif);;PNG Images (*.png);;All Files (*)",
        )

        if not path:
            return

        try:
            export_image = ConductorImageProcessor.normalize_to_uint8(self.display_image)

            if path.lower().endswith(".png"):
                cv2.imwrite(path, export_image)
            else:
                tiff.imwrite(path, export_image)

            self.status_label.setText(f"Exported processed image: {path}")

        except Exception as exc:
            self.status_label.setText(f"Failed to export image: {exc}")


def main():
    app = QApplication(sys.argv)
    window = BRPConductorViewer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
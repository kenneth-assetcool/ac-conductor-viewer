# BRP Coating Coverage Inspector

Python desktop tool for viewing long stitched conductor TIFF images and estimating coating coverage from robotic coating platform captures.

The tool is intended for internal engineering use while developing, validating, and improving coating inspection workflows for BRP robotic platforms.

## Purpose

The BRP coating module captures imagery of overhead conductors while the robotic platform traverses the line. These captures can be stitched into long TIFF images representing the conductor surface over the acquisition length.

This tool helps engineers:

- Open and inspect long TIFF / BigTIFF conductor images.
- Pan, zoom, and auto-pan along the conductor length.
- Preserve vertical image detail while downsampling only along the conductor length.
- Estimate coating coverage for white coating.
- Estimate coating coverage for dark gray coating.
- Exclude black background/gap regions from coverage calculations.
- Display coating masks and coating overlays.
- Calculate visible-window and full-preview coating coverage.
- Export coating coverage profiles as CSV files for further analysis.

## Current Features

### Image Viewing

- Open long stitched conductor TIFF images.
- X-only preview downsampling for very wide images.
- Full vertical resolution is preserved so stacked camera strips remain readable.
- Manual pan and zoom.
- Auto-pan along the conductor length.
- Jump to an original X pixel coordinate.
- Adjustable viewing window width.
- Overview mode for seeing the full preview image.

### Image Processing Modes

The current tool includes the following display modes:

- Original
- CLAHE Local Contrast
- Flatten Background
- Sobel Edges
- Laplacian Detail
- Sharpen
- Dark Impurities
- Bright Scratches
- White Coating Mask
- White Coating Overlay
- Dark Gray Coating Mask
- Dark Gray Coating Overlay
- Valid Conductor Area

### Coating Coverage Tools

The coating quantification tools currently support:

- White coating detection using brightness, saturation, and valid-area thresholds.
- Dark gray coating detection using dark-min, dark-max, saturation, and valid-area thresholds.
- Visible-window coverage calculation.
- Full-preview coverage calculation.
- Coverage profile export along the conductor length.

The coverage calculation is based on:

```text
coverage (%) = coated pixels / valid conductor pixels × 100
```

Black gaps and background areas are excluded using the valid conductor area mask.

## Repository Structure

Suggested repository structure:

```text
brp-coating-inspector/
├── brp_coating_inspector.py
├── brp_conductor_viewer.py
├── README.md
├── requirements.txt
└── .gitignore
```

`brp_coating_inspector.py` is the main tool for coating coverage analysis.

`brp_conductor_viewer.py` can be kept as an earlier/simple viewer if useful, or removed later once the coating inspector becomes the main application.

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:YOUR_ORG_OR_USERNAME/brp-coating-inspector.git
cd brp-coating-inspector
```

Replace `YOUR_ORG_OR_USERNAME` with the correct GitHub organisation or user account.

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with:

```text
PySide6
pyqtgraph
tifffile
opencv-python
scikit-image
numpy
```

## Running the Tool

With the virtual environment activated:

```bash
python brp_coating_inspector.py
```

## Recommended First Use

1. Run the tool.
2. Click **Open TIF**.
3. Select a stitched conductor TIFF image.
4. Start with **Original** mode to inspect the image.
5. Use **Reset View** to return to an inspection window.
6. Use **Auto Pan** to review the conductor length.
7. Select **White Coating Overlay** or **Dark Gray Coating Overlay**.
8. Tune the threshold sliders.
9. Review the visible coverage percentage in the status bar.
10. Export a coverage CSV if needed.

## Suggested Starting Settings

### White Coating

For white coating on silver/light-gray conductor, start with:

```text
Mode: White Coating Overlay
White brightness: 160 to 210
Max saturation: 60 to 110
Background: 15 to 30
```

White coating and bare aluminium can both be bright, so the threshold values may need tuning depending on lighting, exposure, and coating thickness.

### Dark Gray Coating

For dark gray coating, start with:

```text
Mode: Dark Gray Coating Overlay
Dark min: 25 to 50
Dark max: 100 to 170
Max saturation: 60 to 120
Background: 15 to 30
```

Dark gray coating is usually easier to separate from bare silver conductor than white coating, but it can be affected by shadows, camera exposure, and black gaps in the stitched image.

## Controls

### Main Controls

| Control | Purpose |
|---|---|
| Open TIF | Load a stitched conductor TIFF image |
| Reset View | Return to an inspection window |
| Overview | Display the full loaded preview |
| Zoom In | Zoom into the current view |
| Zoom Out | Zoom out from the current view |
| Mode | Select viewing, processing, or coating analysis mode |
| Auto Pan | Automatically move along the conductor length |
| Pan Speed | Control auto-pan speed |
| View Width | Set the width of the inspection window |
| Jump to Original X | Jump to a known original pixel coordinate |
| Export Current Image | Export the current processed image/mask/overlay |
| Export Coverage CSV | Export coating coverage profile along X |

### Coating Controls

| Control | Purpose |
|---|---|
| White Brightness | Minimum brightness for white coating detection |
| Max Saturation | Maximum saturation for neutral coating detection |
| Dark Min | Minimum brightness for dark gray coating detection |
| Dark Max | Maximum brightness for dark gray coating detection |
| Background | Threshold used to exclude black gaps/background |
| CSV Bin Width | Slice width used when exporting coverage profile |

## Coverage CSV Export

The CSV export calculates coating coverage in vertical slices along the conductor length.

Example output columns:

```text
preview_x_start
preview_x_end
original_x_start
original_x_end
coverage_percent
coated_pixels
valid_pixels
uncoated_pixels
```

This allows engineers to plot coating coverage along the conductor length and identify regions with reduced coating coverage.

## Important Notes About Large Files

Large conductor TIFFs can be very wide. For example:

```text
456,496 × 2,385 px
```

The tool avoids displaying the entire full-resolution image at once by downsampling only along the X direction. This keeps the camera-strip height readable while reducing memory usage.

The current tool works as a practical engineering viewer, but future versions should ideally support tiled full-resolution viewing.

## Data Handling

Do not commit raw capture data to the repository.

Avoid committing:

- `.mkv`
- `.mp4`
- `.avi`
- `.mov`
- `.tif`
- `.tiff`
- `.csv`
- `.zarr`
- `.ome.zarr`

Keep large image/video data in a separate shared storage location.

## Suggested `.gitignore`

```text
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
.venv/
venv/
env/

# macOS
.DS_Store

# IDEs
.vscode/
.idea/

# Data files - do not commit large captures
*.mkv
*.mp4
*.avi
*.mov
*.tif
*.tiff
*.csv
*.zarr/
*.ome.zarr/

# Build outputs
build/
dist/
*.spec
```

## Known Limitations

- Current coating detection is threshold-based.
- White coating detection can be challenging because bare conductor is also silver/light gray.
- Lighting, exposure, reflection, shadows, and camera differences can affect results.
- Current coverage results are based on the loaded preview image, not full-resolution tiled analysis.
- The tool does not yet use calibrated real-world distance units.
- The tool does not yet separate the three camera strips into independent lanes.
- The tool does not yet track conductor movement directly from MKV files.
- The tool does not yet perform automatic defect classification.

## Recommended Next Improvements

### 1. Full-Resolution ROI Inspection

Use the preview to navigate, then load a selected region from the original TIFF at full resolution.

### 2. Camera Strip Mode

Display the three camera strips as separate synchronized lanes:

```text
Camera 1
Camera 2
Camera 3
```

This would make coating coverage inspection clearer.

### 3. Calibration-Based Coating Detection

Allow the user to click sample regions:

- Bare conductor sample
- Coated conductor sample
- Background sample

Then classify pixels based on calibrated colour and texture distance rather than fixed thresholds.

### 4. Coverage per Physical Distance

Map pixels to real-world distance using encoder data or stitching metadata.

Example output:

```text
Coverage from 0.0 m to 0.5 m: 94.2 %
Coverage from 0.5 m to 1.0 m: 91.7 %
```

### 5. Defect Detection

Add detection for:

- Missed coating
- Patchy coating
- Scratches
- Surface contamination
- Dark impurities
- Bright marks
- Streaks along conductor length

### 6. Report Generation

Generate a PDF or HTML report including:

- Overall coverage
- Coverage profile
- Defect thumbnails
- Position along conductor
- Representative images
- Detection settings used

## Development Workflow

Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

Commit changes:

```bash
git add .
git commit -m "Describe the change"
```

Push the branch:

```bash
git push -u origin feature/your-feature-name
```

Open a pull request on GitHub for review.

## Author

Arpys Arevalo  
AssetCool

## Status

Early internal engineering tool for collaborative development and validation.
# ac-conductor-viewer

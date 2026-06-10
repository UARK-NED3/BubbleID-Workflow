# BubbleID Workflow Toolkit

BubbleID Workflow Toolkit is a reproducible workflow layer for [BubbleID](https://github.com/cldunlap73/BubbleID), the computer-vision framework for boiling bubble dynamics analysis.

BubbleID remains the scientific engine for segmentation, classification, tracking, vapor fraction, departure rate, bubble statistics, and interface velocity. This toolkit handles the surrounding research workflow: project checks, reproducible configs, still-image segmentation runs, output inspection, and report drafting.

## Why This Repo Is Separate

Keeping the workflow toolkit separate protects the core BubbleID package from fast-changing automation, demo, and lab-specific orchestration. This repo can evolve quickly while BubbleID stays focused on validated computer-vision analysis.

## Current Capabilities

- Validate a BubbleID project before analysis.
- Warn when metadata are missing, such as frame rate or pixel calibration.
- Warn when frame rate is low for tracking-based dynamics.
- Segment still images with BubbleID/Detectron2 instance-segmentation weights and compute vapor fraction.
- Run BubbleID through its `DataAnalysis` class and write a manifest.
- Inspect expected BubbleID outputs for missing files.
- Draft a Markdown report from the manifest and inspection results.
- Fall back to deterministic local report generation when no OpenAI API key is available.

## Quick Demo

For a student-friendly still-image demo on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_selected_images_demo.ps1
```

The script downloads demo images and pretrained one-class BubbleID weights from the repo release, installs the needed CV dependencies in `.venv-bubbleid-demo`, runs segmentation, computes vapor fraction, and writes overlays plus a CSV file. See [docs/quick-demo.md](docs/quick-demo.md) for the full walkthrough.

## Installation

```bash
git clone https://github.com/UARK-NED3/BubbleID-Workflow.git
cd BubbleID-Workflow
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev]
```

For full BubbleID analysis, install BubbleID and its CV dependencies following the BubbleID README. The `--dry-run`, inspection, and offline reporting paths work without importing BubbleID.

For still-image segmentation with pretrained BubbleID Mask R-CNN weights, install PyTorch, OpenCV, and Detectron2 in the environment. On Windows, Detectron2 usually needs Visual Studio Build Tools and may need to be built from source:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install opencv-python matplotlib pycocotools fvcore iopath yacs
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
```

If building from a Visual Studio developer prompt, set `DISTUTILS_USE_SDK=1` before installing Detectron2.

## Configuration

Create a config interactively:

```bash
bubbleid-workflow init-case configs/case-a.json
```

The wizard asks for your video, frame folder, output folder, BubbleID model weights, device, frame rate, pixel calibration, run ID, and confidence threshold.

You can also copy `examples/config.example.json` and update paths for your experiment:

```json
{
  "video_path": "data/case-a/clip.avi",
  "frames_dir": "data/case-a/frames",
  "output_dir": "outputs/case-a",
  "segmentation_weights": "weights/model_final.pth",
  "classification_weights": "weights/classifier.pth",
  "device": "cpu",
  "frame_rate_fps": 3000,
  "pixel_size_um": 12.5,
  "run_id": "case-a",
  "confidence_threshold": 0.5
}
```

## Commands

Segment still images and compute vapor fraction directly:

```bash
bubbleid-workflow segment-images data/selected-images weights/model_1class.pth outputs/selected-images --threshold 0.4 --device cpu
```

By default, still-image segmentation applies an aggressive post-processing step that removes dark, neutral lower-substrate pixels from predicted masks. This reduces slab inclusion in the demo case, but may remove some real lower bubble pixels. Use `--substrate-filter-strength conservative` for a lighter cleanup, or `--no-substrate-filter` to keep raw model masks.

For datasets with matching beginning-of-test substrate images, pass `--substrate-references-dir` with Flat/MP/MC reference images. The selected-images demo uses the reference images in `examples/substrate-references` to remove additional lower substrate components.

Create a reusable still-image case config:

```bash
bubbleid-workflow init-image-case configs/my-image-case.json
```

Run that config:

```bash
bubbleid-workflow run-image-case configs/my-image-case.json
```

Create a case config:

```bash
bubbleid-workflow init-case configs/case-a.json
```

Check inputs:

```bash
bubbleid-workflow check-project configs/case-a.json
```

Write a manifest without invoking BubbleID:

```bash
bubbleid-workflow run-analysis configs/case-a.json --dry-run
```

Run BubbleID:

```bash
bubbleid-workflow run-analysis configs/case-a.json
```

Inspect outputs:

```bash
bubbleid-workflow inspect-outputs outputs/case-a --extension case-a
```

Write a report without an API call:

```bash
bubbleid-workflow write-report outputs/case-a/manifest_case-a.json outputs/case-a reports/case-a.md --extension case-a --offline
```

Write a report with OpenAI:

```bash
bubbleid-workflow write-report outputs/case-a/manifest_case-a.json outputs/case-a reports/case-a.md --extension case-a
```

## OpenAI Setup

Set `OPENAI_API_KEY` in `.env.local` or your shell environment. You can choose a report model with:

```bash
BUBBLEID_AGENT_MODEL=gpt-5
```

Do not commit `.env.local` or any API keys.

## Research Notes

BubbleID Workflow Toolkit does not decide whether BubbleID results are scientifically valid. It flags workflow risks that a boiling researcher should inspect, such as low frame rate for tracking, missing calibration, missing output files, or incomplete model outputs. Physical claims about CHF, interface velocity, departure dynamics, or surface comparisons should still be grounded in experiment metadata, uncertainty, and visual review of segmentation/tracking results.

For still images, `segment-images` reports vapor fraction as the union of predicted bubble-mask pixels divided by total image pixels. This supports frame-wise quantities such as vapor fraction and detected bubble count; it does not produce tracking, departure frequency, or interface velocity because those require time-resolved image sequences.

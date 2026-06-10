# Quick Demo: Segment Boiling Images And Compute Vapor Fraction

This demo is designed for students who are experimentalists and have limited machine-learning setup experience. It uses 24 selected boiling images and BubbleID pretrained one-class instance-segmentation weights.

The demo does four things:

1. Downloads selected demo images and `model_1class.pth` from the BubbleID Workflow GitHub release.
2. Runs BubbleID Workflow still-image segmentation.
3. Computes vapor fraction for each image.
4. Saves overlays, masks, a CSV table, a summary JSON file, and a contact sheet.

## Windows Quick Start

Open PowerShell in the BubbleID Workflow repo:

```powershell
git clone https://github.com/UARK-NED3/BubbleID-Workflow.git
cd BubbleID-Workflow
powershell -ExecutionPolicy Bypass -File .\scripts\run_selected_images_demo.ps1
```

The script creates a local `.venv-bubbleid-demo` environment, installs PyTorch/OpenCV/Detectron2, downloads the demo assets, runs segmentation, and prints the output paths.

If Detectron2 installation fails, install Microsoft C++ Build Tools with the C++ compiler workload, then rerun:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_selected_images_demo.ps1
```

## Outputs

By default, outputs are written under:

```text
.demo-data/selected-images-demo/outputs/
```

Important files:

- `vapor_fraction_results.csv`: one row per image.
- `summary.json`: aggregate vapor-fraction range and run metadata.
- `masks/`: binary vapor-mask images.
- `overlays/`: original images with predicted masks, contours, boxes, and scores.
- `overlay_contact_sheet.jpg`: a quick visual review sheet.

The demo applies an aggressive substrate filter to reduce cases where the black lower slab is counted as vapor. This may remove some real lower bubble pixels, so review the overlays. For a lighter cleanup, rerun `bubbleid-workflow segment-images` with `--substrate-filter-strength conservative`; for raw model masks, use `--no-substrate-filter`.

## What The Numbers Mean

For each still image, BubbleID Workflow reports:

```text
vapor_fraction = union(predicted bubble mask pixels) / total image pixels
```

This is a frame-wise vapor-area fraction. It is not departure frequency, interface velocity, or tracking continuity. Those quantities require a video or ordered high-speed frame sequence.

## Run The Demo Again Without Reinstalling

After the first successful run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_selected_images_demo.ps1 -SkipInstall
```

## Use Your Own Images

Once the demo works, replace the image folder and weights in the command:

```powershell
.\.venv-bubbleid-demo\Scripts\bubbleid-workflow.exe init-image-case configs\my-case.json
.\.venv-bubbleid-demo\Scripts\bubbleid-workflow.exe run-image-case configs\my-case.json
```

Review the overlays before using the CSV in a report, thesis, or publication.

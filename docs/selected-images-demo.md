# Selected Images Demo Workflow

This demo shows how to use BubbleID Workflow with BubbleID pretrained instance-segmentation weights to compute vapor fraction from still boiling images.

The command expects:

- A folder of still images (`.jpg`, `.png`, `.tif`, `.bmp`, and related extensions).
- BubbleID instance-segmentation weights, typically `model_1class.pth` for BubbleID `>=0.0.8`.
- A local Python environment with PyTorch, OpenCV, and Detectron2 installed.

Run:

```bash
bubbleid-workflow segment-images data/selected-images weights/model_1class.pth outputs/selected-images --threshold 0.4 --device cpu
```

Outputs:

- `vapor_fraction_results.csv`: one row per image with image size, bubble-mask pixels, vapor fraction, detected bubble count, and prediction-score summary.
- `summary.json`: aggregate run metadata and vapor-fraction range.
- `masks/`: binary mask PNGs from the union of predicted bubble instances.
- `overlays/`: image overlays with predicted masks, contours, bounding boxes, and scores.

Interpretation:

Vapor fraction is computed as:

```text
vapor_fraction = union(predicted bubble mask pixels) / total image pixels
```

For still images, this is a frame-wise vapor-area estimate. It should not be interpreted as departure frequency, interface velocity, or tracking continuity. Those quantities require a time-resolved video or ordered frame sequence.

Quality checks:

- Review the overlay images before using the CSV in a paper, thesis, or report.
- Small faint bubbles may be missed by the pretrained model.
- Very large CHF/NBR vapor structures may dominate the image and produce high vapor-fraction values.
- If the camera setup, pressure, surface, lighting, or fluid differs substantially from BubbleID training data, fine-tuning with annotated images may improve reliability.

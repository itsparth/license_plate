#!/usr/bin/env python3
"""Export RF-DETR models to ONNX format.

Exports both license plate detection and character detection models.
Uses legacy torch.onnx.export for compatibility with PyTorch 2.9.
"""

from pathlib import Path

import torch

from rfdetr.detr import RFDETRNano

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"
LP_DETECTION_MODEL = OUTPUT_DIR / "lp_detection_training" / "checkpoint_best_ema.pth"
CHAR_DETECTION_MODEL = OUTPUT_DIR / "rfdetr_training" / "checkpoint_best_ema.pth"
EXPORT_DIR = OUTPUT_DIR / "exported_models"


def export_model(checkpoint_path: Path, output_path: Path) -> None:
    """Export a single model to ONNX format using legacy exporter."""
    print(f"Loading model from {checkpoint_path}")
    rfdetr = RFDETRNano(pretrain_weights=str(checkpoint_path))

    # Get the actual PyTorch model (rfdetr.model is wrapper, rfdetr.model.model is LWDETR)
    model = rfdetr.model.model  # type: ignore[union-attr]

    # Move to CPU FIRST before export mode
    model.cpu()  # type: ignore[union-attr]
    model.eval()  # type: ignore[union-attr]

    # Prepare model for export (switches to export-friendly forward that returns tuple)
    if hasattr(model, "export"):
        model.export()  # type: ignore[union-attr]

    # RFDETRNano uses 384x384 input
    input_size = rfdetr.model.resolution
    device = torch.device("cpu")

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # Run inference to check output shapes (export mode returns tuple: boxes, logits, None)
    with torch.no_grad():
        boxes, logits, _ = model(dummy_input)  # type: ignore[operator]
        print(f"Output shapes - Boxes: {boxes.shape}, Logits: {logits.shape}")

    print(f"Exporting to {output_path}")
    # Use dynamo=False to force legacy TorchScript-based ONNX export
    # Output names match DeepStream-rfdetr: dets (boxes), labels (logits)
    torch.onnx.export(
        model,  # type: ignore[arg-type]
        (dummy_input,),
        str(output_path),
        input_names=["input"],
        output_names=["dets", "labels"],
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        opset_version=18,
        dynamo=False,
        verbose=False,
    )
    print(f"Exported: {output_path.name}")


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    models = [
        (LP_DETECTION_MODEL, EXPORT_DIR / "lp_detection.onnx"),
        (CHAR_DETECTION_MODEL, EXPORT_DIR / "char_detection.onnx"),
    ]

    for checkpoint, output_path in models:
        if not checkpoint.exists():
            print(f"Checkpoint not found: {checkpoint}")
            continue

        try:
            export_model(checkpoint, output_path)
        except Exception as e:
            print(f"Failed to export {output_path.name}: {e}")

    print(f"\nExported models saved to: {EXPORT_DIR}")


if __name__ == "__main__":
    main()

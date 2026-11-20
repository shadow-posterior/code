"""Normalize MMDetection outputs for GenEval's per-class selection logic."""

import numpy as np
from pathlib import Path


def resolve_detector_checkpoint(config_path, checkpoint_root):
    """Select local weights for the config's detector version; never mix versions."""
    name = Path(config_path).name
    if "8xb2-lsj-50e_coco" in name:
        filename = "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
    elif "lsj_8x2_50e_coco" in name:
        filename = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
    else:
        raise ValueError(f"Unknown Mask2Former config: {config_path}")
    root = Path(checkpoint_root).expanduser()
    candidates = [root / filename, root / "geneval" / filename]
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"Missing compatible Mask2Former checkpoint for {name}: {candidates}")


def group_detector_result(result, num_classes):
    """Keep boxes, confidence and masks aligned across MMDetection 2 and 3."""
    if not hasattr(result, "pred_instances"):
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None

    pred = result.pred_instances
    boxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()
    masks = pred.masks.detach().cpu().numpy() if hasattr(pred, "masks") else None
    scored_boxes = np.concatenate((boxes, scores[:, None]), axis=1)
    grouped_boxes = [scored_boxes[labels == index] for index in range(num_classes)]
    grouped_masks = (
        [masks[labels == index] for index in range(num_classes)]
        if masks is not None else None
    )
    return grouped_boxes, grouped_masks

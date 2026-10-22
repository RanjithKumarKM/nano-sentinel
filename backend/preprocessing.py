import cv2
import numpy as np


def preprocess_wafer_map(wafer, target_size=96):
    """Convert a categorical 2D wafer map into a model input."""
    wafer = np.asarray(wafer)

    if wafer.ndim != 2 or wafer.size == 0:
        raise ValueError("Expected a non-empty 2D wafer map.")

    if not np.isfinite(wafer).all():
        raise ValueError("Wafer map contains invalid values.")

    # WM-811K categorical values: 0=background, 1=normal, 2=defective.
    if not np.isin(wafer, [0, 1, 2]).all():
        raise ValueError("Wafer map must contain only values 0, 1, and 2.")

    wafer = wafer.astype(np.uint8)
    height, width = wafer.shape

    scale = target_size / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))

    resized = cv2.resize(
        wafer,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized

    rgb = np.repeat(canvas[..., None], 3, axis=-1).astype(np.float32)

    # Must match the preprocessing used to train the saved model.
    return (rgb / 2.0) * 2.0 - 1.0

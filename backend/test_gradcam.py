
import numpy as np

from gradcam import generate_gradcam

# Synthetic categorical wafer map for a pipeline test only.
wafer = np.zeros((32, 32), dtype=np.uint8)
wafer[4:28, 4:28] = 1
wafer[12:18, 12:18] = 2

result = generate_gradcam(wafer)

print("Predicted class:", result["predicted_class"])
print("Confidence:", result["confidence_percent"])
print("Heatmap shape:", result["heatmap"].shape)
print("Heatmap range:",
      float(result["heatmap"].min()),
      float(result["heatmap"].max()))
print("Target layer:", result["target_layer"])

assert result["heatmap"].shape == wafer.shape
assert np.isfinite(result["heatmap"]).all()
assert result["heatmap"].min() >= 0
assert result["heatmap"].max() <= 1.0001

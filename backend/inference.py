import os
import numpy as np
import tensorflow as tf

from preprocessing import preprocess_wafer_map

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none"
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(
    ROOT, "models", "nano_sentinel_mobilenetv2_v2.keras"
)
MODEL_PATH = os.environ.get("NANO_SENTINEL_MODEL", DEFAULT_MODEL)

_model = None


def get_model():
    """Load the model only once per backend process."""
    global _model

    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. "
                "Copy the .keras model into the project's models folder."
            )
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    return _model


def predict_wafer(wafer):
    processed = preprocess_wafer_map(wafer)
    batch = np.expand_dims(processed, axis=0)

    model = get_model()
    outputs = np.asarray(model.predict(batch, verbose=0))[0]

    if len(outputs) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected 9 model outputs, received {len(outputs)}."
        )

    # Keras classifier outputs are expected to be probabilities.
    # Validate rather than silently applying a second softmax.
    if (
        not np.isfinite(outputs).all()
        or np.any(outputs < 0)
        or not np.isclose(outputs.sum(), 1.0, atol=1e-3)
    ):
        raise ValueError(
            "Model outputs are not valid probabilities. "
            "Check the saved model's final activation."
        )

    best_index = int(np.argmax(outputs))

    return {
        "predicted_class": CLASS_NAMES[best_index],
        "confidence_percent": round(float(outputs[best_index]) * 100, 2),
        "probabilities": {
            name: round(float(value), 8)
            for name, value in zip(CLASS_NAMES, outputs)
        }
    }

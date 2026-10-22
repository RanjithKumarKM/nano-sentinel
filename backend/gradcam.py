
import numpy as np
import tensorflow as tf

from inference import get_model, CLASS_NAMES
from preprocessing import preprocess_wafer_map


BACKBONE_NAME = "mobilenetv2_1.00_96"
TARGET_LAYER_NAME = "block_13_expand"


def generate_gradcam(wafer):
    model = get_model()
    backbone = model.get_layer(BACKBONE_NAME)
    target_layer = backbone.get_layer(TARGET_LAYER_NAME)

    processed = preprocess_wafer_map(wafer)
    batch = np.expand_dims(processed, axis=0).astype(np.float32)

    # Extract intermediate convolutional features and final backbone features.
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            target_layer.output,
            backbone.output
        ],
        name="nano_sentinel_gradcam_backbone"
    )

    # Reuse the trained classification head and its existing weights.
    head_input = tf.keras.Input(shape=backbone.output_shape[1:])

    x = model.get_layer("global_average_pooling2d_1")(head_input)
    x = model.get_layer("dropout_1")(x)
    head_output = model.get_layer("dense_1")(x)

    classification_head = tf.keras.Model(
        inputs=head_input,
        outputs=head_output,
        name="nano_sentinel_gradcam_head"
    )

    with tf.GradientTape() as tape:
        conv_features, backbone_features = backbone_grad_model(
            [batch],
            training=False
        )

        predictions = classification_head(
            backbone_features,
            training=False
        )

        class_index = tf.argmax(predictions[0])
        target_score = predictions[:, class_index]

    gradients = tape.gradient(target_score, conv_features)

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients are unavailable for the selected layer."
        )

    # Average gradients to obtain the importance of each feature channel.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    feature_map = conv_features[0]

    heatmap = tf.reduce_sum(
        feature_map * channel_weights,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, maximum)

    # Resize for display; this does not increase the map's true resolution.
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        size=(int(wafer.shape[0]), int(wafer.shape[1])),
        method="bilinear"
    )[..., 0]

    probabilities = predictions[0].numpy()
    predicted_index = int(class_index.numpy())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence_percent": round(
            float(probabilities[predicted_index]) * 100, 2
        ),
        "heatmap": heatmap.numpy(),
        "target_layer": TARGET_LAYER_NAME
    }

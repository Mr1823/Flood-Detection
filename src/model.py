"""
model.py - MobileNetV2 transfer-learning model: flood vs no_flood.

The whole network is ONE flat functional model:

    image (raw RGB, 0..255)
      -> augmentation            random flip / rotate / zoom / brightness / contrast
                                 (active only while training)
      -> mobilenetv2_preprocess  mobilenet_v2.preprocess_input: 0..255 -> -1..1
                                 *** the ONLY place preprocessing happens (see dataset.py) ***
      -> MobileNetV2 body        ImageNet weights, frozen. Built with input_tensor=, so its
                                 layers - including 'Conv_1' for Grad-CAM - belong to THIS
                                 model instead of hiding inside a nested sub-model
      -> GlobalAveragePooling2D -> Dropout -> Dense(relu) -> Dropout
      -> Dense(1, sigmoid)       = probability that the image shows a flood
"""
import os

import numpy as np

import config  # imported before TensorFlow so its log-level setting takes effect
import tensorflow as tf

layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package="flood_detection")
class MobileNetV2Preprocess(layers.Layer):
    """mobilenet_v2.preprocess_input as a layer: raw pixels 0..255 -> -1..1.

    MobileNetV2 was trained on ImageNet images scaled to -1..1, so it has to see
    the same range here. (Rescaling(1./255) would give 0..1 - the classic mistake.)
    As a registered layer it is saved inside best_model.keras and the TFLite file,
    so no script that uses the model can forget it or apply it twice.
    """

    def call(self, inputs):
        return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)


def _assert_flat(model):
    """Grad-CAM needs model.get_layer('Conv_1'): fail now rather than at evaluation time."""
    try:
        model.get_layer(config.GRADCAM_LAYER)
    except ValueError as exc:
        raise RuntimeError(
            f"Layer '{config.GRADCAM_LAYER}' is not reachable from the top-level model: "
            "MobileNetV2 has ended up nested as a sub-model, so Grad-CAM cannot reach its "
            "feature maps. Build the base with input_tensor=... (see build_model).") from exc


def build_model(img_size=config.IMG_SIZE, channels=config.CHANNELS, dropout_1=config.DROPOUT_1,
                dense_units=config.DENSE_UNITS, dropout_2=config.DROPOUT_2, augmentation=None):
    """Build (not compile) the classifier with a frozen MobileNetV2 backbone."""
    inputs = layers.Input(shape=(*img_size, channels), name="image")
    x = augmentation(inputs) if augmentation is not None else inputs
    x = MobileNetV2Preprocess(name="mobilenetv2_preprocess")(x)

    # input_tensor= makes MobileNetV2 build its layers directly on top of x, so they
    # become layers of THIS model. Writing base(x) instead would wrap the whole
    # network as a single nested layer, and get_layer("Conv_1") would fail.
    # input_shape is stated too, so Keras checks the tensor really is 224x224x3.
    base = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                                             input_tensor=x, input_shape=(*img_size, channels))
    base.trainable = False    # phase 1: keep the ImageNet features fixed. A frozen BatchNorm
                              # layer also runs in inference mode (uses its stored statistics).

    y = layers.GlobalAveragePooling2D(name="gap")(base.output)   # 7x7x1280 maps -> 1280 numbers
    y = layers.Dropout(dropout_1, name="dropout_1")(y)
    y = layers.Dense(dense_units, activation="relu", name="dense")(y)
    y = layers.Dropout(dropout_2, name="dropout_2")(y)
    outputs = layers.Dense(1, activation="sigmoid", name="flood_probability")(y)
    model = tf.keras.Model(inputs, outputs, name="flood_mobilenetv2")

    _assert_flat(model)
    return model


def count_trainable_params(model):
    """Number of individual weights the optimiser is allowed to change."""
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def unfreeze_from(model, layer_name=config.UNFREEZE_FROM):
    """Phase 2: make every layer from `layer_name` onward trainable - EXCEPT BatchNormalization.

    BatchNorm layers stay trainable=False, which also keeps them in inference mode
    with ImageNet's stored mean and variance. Letting them re-estimate those from
    our small batches would wreck the pretrained features.
    The model must be recompiled afterwards for the change to take effect.
    """
    names = [layer.name for layer in model.layers]
    if layer_name not in names:
        raise ValueError(f"No layer named '{layer_name}' in the model "
                         "(expected a MobileNetV2 layer name such as 'block_13_expand').")
    start = names.index(layer_name)
    before = count_trainable_params(model)

    unfrozen, unfrozen_with_weights, bn_frozen = 0, 0, 0
    for layer in model.layers[start:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            bn_frozen += 1
        elif not layer.trainable:
            layer.trainable = True
            unfrozen += 1
            unfrozen_with_weights += bool(layer.weights)

    print(f"[model] Unfroze {unfrozen} layers from '{layer_name}' onward "
          f"({unfrozen_with_weights} of them hold weights); {bn_frozen} BatchNorm layers kept frozen.")
    print(f"[model] Trainable parameters: {before:,} -> {count_trainable_params(model):,}")
    return model


def compile_model(model, learning_rate):
    """Adam + binary cross-entropy, tracking accuracy, precision, recall and AUC.

    Precision, recall and accuracy use the default 0.5 threshold here; AUC does
    not depend on any threshold, which is why it is used to pick the best model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def check_device_consistency(model, images, tolerance=1e-3):
    """Guard against wrong numbers from the GPU plugin: the same batch must give the same
    probabilities in graph mode on the default device (the GPU, as fit/evaluate/predict
    run) and eagerly on the CPU (the reference). Fails loudly if they differ."""
    images = tf.convert_to_tensor(np.asarray(images, dtype=np.float32))
    graph = tf.function(lambda x: model(x, training=False))(images).numpy()
    with tf.device("/CPU:0"):
        reference = model(images, training=False).numpy()
    gap = float(np.abs(graph - reference).max())
    if gap > tolerance:
        raise RuntimeError(
            f"Device check FAILED: graph-mode output differs from the CPU reference by {gap:.3g} "
            f"(tolerance {tolerance}). The GPU plugin is computing wrong results - see the "
            "use_plugin_optimizers note in config.py. Do not trust any metric from this setup.")
    print(f"[model] Device check passed: GPU graph mode matches the CPU reference "
          f"(max difference {gap:.1e} on {len(graph)} images)")
    return gap


def load_trained_model(path=None, compile=True):
    """Load a saved model (default models/best_model.keras), failing loudly if it is missing.

    Importing this module registers MobileNetV2Preprocess, which load_model needs.
    """
    path = path or os.path.join(config.MODEL_DIR, config.BEST_MODEL_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No trained model at {config.rel(path)}. "
                                "Train one first:  python src/train.py")
    model = tf.keras.models.load_model(path, compile=compile)
    _assert_flat(model)
    return model

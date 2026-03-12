import tensorflow as tf
import os
import logging

logger = logging.getLogger(__name__)

def convert_savedmodel_to_tflite(saved_model_dir, tflite_model_path):

    if not os.path.exists(saved_model_dir):
        logger.error(f"Cannot find SavedModel at {saved_model_dir}")
        return False

    logger.info(f"Loading SavedModel from {saved_model_dir}...")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Optimization (size + latency improvements)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Allow float16 for better inference in kotlin app
    converter.target_spec.supported_types = [tf.float16]

    # Allow TF ops if your GNN layer needs them
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    logger.info(f"TFLite model saved to: {tflite_model_path}")

    # Verify model loads
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    signatures = interpreter.get_signature_list()

    logger.info(f"TFLite Signatures: {signatures}")

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    convert_savedmodel_to_tflite(
        "models/gnn_model",
        "models/gnn_model.tflite"
    )
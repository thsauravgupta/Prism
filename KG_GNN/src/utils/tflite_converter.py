import tensorflow as tf
import os
import logging

logger = logging.getLogger(__name__)

def convert_h5_to_tflite(h5_model_path, tflite_model_path):
    """
    Converts a Keras .h5 model to optimize TFLite format for the final on-device Kotlin app.
    """
    if not os.path.exists(h5_model_path):
        logger.error(f"Cannot find model file at {h5_model_path}")
        return False
        
    logger.info(f"Loading model from {h5_model_path} for TFLite conversion...")
    
    # We must provide custom objects if we used them in keras but since
    # our GCNEmbeddingLayer is standard, we can load it from the source if needed.
    # To safely load the model, we import the layer:
    from src.models.gnn_model import GCNEmbeddingLayer
    
    custom_objects = {"GCNEmbeddingLayer": GCNEmbeddingLayer}
    model = tf.keras.models.load_model(h5_model_path, custom_objects=custom_objects)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations like quantization for smaller size on device
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: ensure ops are supported smoothly on typical mobile hardware
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # Enable TensorFlow ops.
    ]

    tflite_model = converter.convert()
    
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        
    logger.info(f"Successfully converted to TFLite! Saved at: {tflite_model_path}")
    
    # Verify inference signature via interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    signatures = interpreter.get_signature_list()
    logger.info(f"TFLite Signatures: {signatures}")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    convert_h5_to_tflite("models/gnn_model.h5", "models/gnn_model.tflite")

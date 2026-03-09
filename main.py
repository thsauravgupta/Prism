import logging
import argparse
from src.training.train import ModelTrainer
from src.utils.tflite_converter import convert_h5_to_tflite

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="SmartThings On-Device Predictive Model Pipeline")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    logger.info("Initializing Pipeline...")
    
    # 1. Train the model (this inherently calls DatasetLoader and KGBuilder)
    trainer = ModelTrainer(data_dir="data", model_dir="models")
    model_path = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    
    # 2. Convert to TFLite
    if model_path:
        tflite_path = model_path.replace(".h5", ".tflite")
        logger.info(f"Converting trained model to TFLite: {tflite_path}")
        convert_h5_to_tflite(model_path, tflite_path)
    else:
        logger.error("Model training failed. Skipping TFLite conversion.")

if __name__ == "__main__":
    main()

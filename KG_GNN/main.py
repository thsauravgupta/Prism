import logging
import argparse
import os
from src.training.train import ModelTrainer
from src.utils.tflite_converter import convert_savedmodel_to_tflite

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="SmartThings On-Device Predictive Model Pipeline"
    )

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    logger.info("Initializing Pipeline...")

    # 1. Train the model
    trainer = ModelTrainer(data_dir="data", model_dir="models")

    model_path = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # 2. Convert to TFLite
    if model_path:

        tflite_path = os.path.join(
            os.path.dirname(model_path),
            "gnn_model.tflite"
        )

        logger.info(f"Converting trained model to TFLite: {tflite_path}")

        convert_savedmodel_to_tflite(
            model_path,
            tflite_path
        )

    else:
        logger.error("Model training failed. Skipping TFLite conversion.")


if __name__ == "__main__":
    main()
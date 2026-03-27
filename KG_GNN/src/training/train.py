# import os
# import tensorflow as tf
# import logging

# from src.data.dictionary_manager import DictionaryManager
# from src.data.dataset_loader import DatasetLoader
# from src.graph.kg_builder import KGBuilder
# from src.models.gnn_model import build_predictive_model

# logger = logging.getLogger(__name__)

# class ModelTrainer:
#     def __init__(self, data_dir="data", model_dir="models"):
#         self.data_dir = data_dir
#         self.model_dir = model_dir
#         os.makedirs(self.model_dir, exist_ok=True)
        
#         self.dictionary_manager = DictionaryManager(data_dir=self.data_dir)
#         self.dataset_loader = DatasetLoader(self.dictionary_manager, data_dir=self.data_dir)
#         self.kg_builder = KGBuilder(self.dictionary_manager, data_dir=self.data_dir)

#     def train(self, epochs=5, batch_size=32):
#         logger.info("Starting Data Loading...")
#         (X_train, y_train), (X_val, y_val), (X_test, y_test), global_dict = self.dataset_loader.load_and_preprocess()
        
#         if len(X_train) == 0:
#             logger.error("No training data found. Make sure valid region data is in the directory.")
#             return None
            
#         logger.info("Building Knowledge Graph Adjacency Matrix...")
#         adj_matrix = self.kg_builder.build_adjacency_matrix()
        
#         logger.info("Building Model...")
#         model = build_predictive_model(
#             global_dicts=self.dictionary_manager.global_dicts,
#             adj_matrix=adj_matrix,
#             seq_length=9
#         )
        
#         # Prepare tf.data.Dataset
#         def prepare_features(X):
#             return {
#                 "dayofweek": X[:, :, 0],
#                 "hour": X[:, :, 1],
#                 "device": X[:, :, 2],
#                 "unknown": X[:, :, 3],
#                 "device_control": X[:, :, 4]
#             }
            
#         train_features = prepare_features(X_train)
#         val_features = prepare_features(X_val)
        
#         # Callbacks
#         callbacks = [
#             tf.keras.callbacks.EarlyStopping(
#                 monitor="val_loss",
#                 patience=3,
#                 restore_best_weights=True
#             ),
#             tf.keras.callbacks.ModelCheckpoint(
#                 filepath=os.path.join(self.model_dir, "best_gnn_model.keras"),
#                 monitor="val_loss",
#                 save_best_only=True,
#                 verbose=1
#             )
#         ]
        
#         logger.info("Starting Training...")
#         self.history = model.fit(
#             train_features, y_train,
#             validation_data=(val_features, y_val),
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=callbacks
#         )
        
#         if len(X_test) > 0:
#             logger.info("Evaluating on Test Set...")
#             test_features = prepare_features(X_test)
#             test_loss, test_acc = model.evaluate(test_features, y_test)
#             logger.info(f"Test Accuracy: {test_acc:.4f}")

        
#         # save the model as a folder instead of file since it is safer to load and convert into tf lite
#         # model_path = os.path.join(self.model_dir, "gnn_model") 
#         # model.save(model_path)
        
#         model_path = os.path.join(self.model_dir, "gnn_model")
#         model.export(model_path)
#         logger.info(f"Model saved to {model_path}")
        
#         return model_path

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     trainer = ModelTrainer()
#     trainer.train(epochs=10, batch_size=64)

import os
import time
import logging
import tensorflow as tf

from src.data.dictionary_manager import DictionaryManager
from src.data.dataset_loader import DatasetLoader
from src.graph.kg_builder import KGBuilder
from src.models.gnn_model import build_predictive_model

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.dictionary_manager = DictionaryManager(data_dir=self.data_dir)
        self.dataset_loader = DatasetLoader(self.dictionary_manager, data_dir=self.data_dir)
        self.kg_builder = KGBuilder(self.dictionary_manager, data_dir=self.data_dir)

    @staticmethod
    def _prepare_features(X):
        return {
            "dayofweek": X[:, :, 0].astype("int32"),
            "hour": X[:, :, 1].astype("int32"),
            "device": X[:, :, 2].astype("int32"),
            "unknown": X[:, :, 3].astype("float32"),
            "device_control": X[:, :, 4].astype("int32"),
        }

    def train(self, epochs=10, batch_size=64):
        logger.info("Loading and preprocessing datasets...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = self.dataset_loader.load_and_preprocess()

        if len(X_train) == 0:
            logger.error("No training data found.")
            return None

        logger.info("Building semantic device adjacency...")
        adj_matrix = self.kg_builder.build_adjacency_matrix()

        logger.info("Building predictive model...")
        model = build_predictive_model(
            global_dicts=self.dictionary_manager.global_dicts,
            adj_matrix=adj_matrix,
            seq_length=9,
            embed_dim_device=8,
            embed_dim_day=4,
            embed_dim_hour=4,
            embed_dim_devctrl=8,
            rnn_units=32,
            dense_units=32,
            dropout_rate=0.2,
            use_gru=True,
        )

        train_features = self._prepare_features(X_train)
        val_features = self._prepare_features(X_val)

        keras_path = os.path.join(self.model_dir, "best_gnn_model.keras")
        export_path = os.path.join(self.model_dir, "gnn_model")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=keras_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        logger.info("Starting training...")
        start_time = time.perf_counter()
        self.history = model.fit(
            train_features,
            y_train,
            validation_data=(val_features, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        train_time_sec = time.perf_counter() - start_time

        test_metrics = {}
        if len(X_test) > 0:
            logger.info("Evaluating on test set...")
            test_features = self._prepare_features(X_test)
            values = model.evaluate(test_features, y_test, verbose=0)
            test_metrics = dict(zip(model.metrics_names, values))
            logger.info("Test metrics: %s", test_metrics)

        logger.info("Saving final .keras model...")
        model.save(os.path.join(self.model_dir, "final_gnn_model.keras"))

        logger.info("Exporting SavedModel...")
        model.export(export_path)
        logger.info("Exported model to %s", export_path)

        return {
            "history": self.history.history,
            "keras_path": keras_path,
            "final_keras_path": os.path.join(self.model_dir, "final_gnn_model.keras"),
            "export_path": export_path,
            "train_time_sec": train_time_sec,
            "test_metrics": test_metrics,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = ModelTrainer()
    results = trainer.train(epochs=10, batch_size=64)
    print(results)
import numpy as np
import tensorflow as tf


class GCNEmbeddingLayer(tf.keras.layers.Layer):
    """
    Lightweight graph-aware embedding layer.

    Applies one graph propagation step:
        H' = relu(A @ H @ W)
    then gathers node embeddings for input device ids.
    """

    def __init__(self, num_devices, embed_dim, adj_matrix, **kwargs):
        super().__init__(**kwargs)
        self.num_devices = int(num_devices)
        self.embed_dim = int(embed_dim)

        adj_matrix = np.asarray(adj_matrix, dtype=np.float32)
        padded_adj = np.zeros((self.num_devices, self.num_devices), dtype=np.float32)

        rows = min(self.num_devices, adj_matrix.shape[0])
        cols = min(self.num_devices, adj_matrix.shape[1])
        padded_adj[:rows, :cols] = adj_matrix[:rows, :cols]

        self.adj_matrix_np = padded_adj
        self.adj_matrix = tf.constant(self.adj_matrix_np, dtype=tf.float32)

    def build(self, input_shape):
        self.device_embeddings = self.add_weight(
            name="device_embeddings",
            shape=(self.num_devices, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W = self.add_weight(
            name="gcn_weight",
            shape=(self.embed_dim, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        graph_features = tf.matmul(self.adj_matrix, self.device_embeddings)
        graph_features = tf.matmul(graph_features, self.W)
        updated_embeddings = tf.nn.relu(graph_features)

        inputs = tf.cast(inputs, tf.int32)
        return tf.gather(updated_embeddings, inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_devices": self.num_devices,
                "embed_dim": self.embed_dim,
                "adj_matrix": self.adj_matrix_np.tolist(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        adj_matrix = np.array(config.pop("adj_matrix"), dtype=np.float32)
        return cls(adj_matrix=adj_matrix, **config)


class ExpandFloatLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.cast(inputs, tf.float32), axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (1,)

    def get_config(self):
        return super().get_config()


def build_predictive_model(
    global_dicts,
    adj_matrix,
    seq_length=9,
    embed_dim_device=8,
    embed_dim_day=4,
    embed_dim_hour=4,
    embed_dim_devctrl=8,
    rnn_units=32,
    dense_units=32,
    dropout_rate=0.2,
    use_gru=True,
):
    vocab_dow = len(global_dicts.get("dayofweek_dict", {})) + 1
    vocab_hr = len(global_dicts.get("hour_dict", {})) + 1
    vocab_dev = len(global_dicts.get("device_dict", {})) + 1
    vocab_ctrl = len(global_dicts.get("device_control_dict", {})) + 1

    in_dow = tf.keras.Input(shape=(seq_length,), dtype=tf.int32, name="dayofweek")
    in_hr = tf.keras.Input(shape=(seq_length,), dtype=tf.int32, name="hour")
    in_dev = tf.keras.Input(shape=(seq_length,), dtype=tf.int32, name="device")
    in_unknown = tf.keras.Input(shape=(seq_length,), dtype=tf.float32, name="unknown")
    in_ctrl = tf.keras.Input(shape=(seq_length,), dtype=tf.int32, name="device_control")

    emb_dow = tf.keras.layers.Embedding(vocab_dow, embed_dim_day, name="emb_day")(in_dow)
    emb_hr = tf.keras.layers.Embedding(vocab_hr, embed_dim_hour, name="emb_hour")(in_hr)
    emb_ctrl = tf.keras.layers.Embedding(vocab_ctrl, embed_dim_devctrl, name="emb_devctrl")(in_ctrl)
    emb_unk = ExpandFloatLayer(name="expand_unknown")(in_unknown)

    gcn_dev = GCNEmbeddingLayer(
        num_devices=vocab_dev,
        embed_dim=embed_dim_device,
        adj_matrix=adj_matrix,
        name="gcn_device_embedding",
    )(in_dev)

    x = tf.keras.layers.Concatenate(axis=-1, name="concat_features")(
        [emb_dow, emb_hr, gcn_dev, emb_ctrl, emb_unk]
    )

    if use_gru:
        x = tf.keras.layers.GRU(rnn_units, return_sequences=False, name="gru")(x)
    else:
        x = tf.keras.layers.LSTM(rnn_units, return_sequences=False, name="lstm")(x)

    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    output = tf.keras.layers.Dense(vocab_dev, activation="softmax", name="prediction")(x)

    model = tf.keras.Model(
        inputs=[in_dow, in_hr, in_dev, in_unknown, in_ctrl],
        outputs=output,
        name="smartthings_predictor",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
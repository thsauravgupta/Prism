import tensorflow as tf

class ExpandFloatLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.cast(inputs, tf.float32), axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (1,)

    def get_config(self):
        return super().get_config()

def build_baseline_gru_model(
    global_dicts,
    seq_length=9,
    embed_dim_day=4,
    embed_dim_hour=4,
    embed_dim_device=8,
    embed_dim_devctrl=8,
    rnn_units=32,
    dense_units=32,
    dropout_rate=0.2,
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

    emb_dow = tf.keras.layers.Embedding(vocab_dow, embed_dim_day)(in_dow)
    emb_hr = tf.keras.layers.Embedding(vocab_hr, embed_dim_hour)(in_hr)
    emb_dev = tf.keras.layers.Embedding(vocab_dev, embed_dim_device)(in_dev)
    emb_ctrl = tf.keras.layers.Embedding(vocab_ctrl, embed_dim_devctrl)(in_ctrl)
    emb_unk = ExpandFloatLayer()(in_unknown)

    x = tf.keras.layers.Concatenate(axis=-1)(
        [emb_dow, emb_hr, emb_dev, emb_ctrl, emb_unk]
    )
    x = tf.keras.layers.GRU(rnn_units, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    output = tf.keras.layers.Dense(vocab_dev, activation="softmax", name="prediction")(x)

    model = tf.keras.Model(
        inputs=[in_dow, in_hr, in_dev, in_unknown, in_ctrl],
        outputs=output,
        name="baseline_gru_model"
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
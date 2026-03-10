import tensorflow as tf
from keras.layers import Input, Embedding, Dense, Concatenate, LSTM, Dropout, Layer
from keras.models import Model
import numpy as np

class GCNEmbeddingLayer(Layer):
    """
    A unified standard Keras layer that stores initial embeddings
    and applies a GCN step using the adjacency matrix.
    """
    def __init__(self, num_devices, embed_dim, adj_matrix, **kwargs):
        super(GCNEmbeddingLayer, self).__init__(**kwargs)
        self.num_devices = num_devices
        self.embed_dim = embed_dim
        # Pad adj_matrix to match num_devices
        n_dev = adj_matrix.shape[0]
        padded_adj = np.zeros((num_devices, num_devices), dtype=np.float32)
        padded_adj[:n_dev, :n_dev] = adj_matrix
        self.adj_matrix = tf.constant(padded_adj, dtype=tf.float32)

    def build(self, input_shape):
        self.device_embeddings = self.add_weight(shape=(self.num_devices, self.embed_dim),
                                                 initializer='uniform',
                                                 trainable=True,
                                                 name='device_embeddings')
        self.W = self.add_weight(shape=(self.embed_dim, self.embed_dim),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='gcn_weight')
        super(GCNEmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        # Apply message passing: H_new = relu(A * H * W)
        graph_features = tf.matmul(self.adj_matrix, self.device_embeddings)
        graph_features = tf.matmul(graph_features, self.W)
        updated_embeddings = tf.nn.relu(graph_features)
        
        # Look up the updated embeddings for the input device IDs
        return tf.gather(updated_embeddings, tf.cast(inputs, tf.int32))

def build_predictive_model(global_dicts, adj_matrix, seq_length=9, embed_dim=16):
    """
    Builds the Keras Deep Learning Model integrating Knowledge Graph (GNN) and sequential data.
    """
    vocab_dow = len(global_dicts.get('dayofweek_dict', {})) + 1
    vocab_hr = len(global_dicts.get('hour_dict', {})) + 1
    vocab_dev = len(global_dicts.get('device_dict', {})) + 1
    vocab_ctrl = len(global_dicts.get('device_control_dict', {})) + 1
    
    in_dow = Input(shape=(seq_length,), name="dayofweek")
    in_hr = Input(shape=(seq_length,), name="hour")
    in_dev = Input(shape=(seq_length,), name="device")
    in_unknown = Input(shape=(seq_length,), name="unknown") 
    in_ctrl = Input(shape=(seq_length,), name="device_control")

    emb_dow = Embedding(input_dim=vocab_dow, output_dim=8)(in_dow)
    emb_hr = Embedding(input_dim=vocab_hr, output_dim=8)(in_hr)
    emb_ctrl = Embedding(input_dim=vocab_ctrl, output_dim=16)(in_ctrl)
    
    # Simple continuous embedding for the unknown column
    emb_unk = tf.expand_dims(tf.cast(in_unknown, tf.float32), -1)

    # GNN-enhanced device embeddings
    gcn_dev = GCNEmbeddingLayer(vocab_dev, embed_dim, adj_matrix)(in_dev)

    # Sequence Processing
    x = Concatenate(axis=-1)([emb_dow, emb_hr, gcn_dev, emb_ctrl, emb_unk])
    
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    
    # Dense projection to probabilities
    x = Dense(32, activation='relu')(x)
    output = Dense(vocab_dev, activation='softmax', name="prediction")(x)

    model = Model(inputs=[in_dow, in_hr, in_dev, in_unknown, in_ctrl], outputs=output)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

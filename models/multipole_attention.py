import tensorflow as tf
from models.attention_cores import MultipoleCore
from models.attention_gates import get_2d_positional_encoding

class MultipoleBottleneck(tf.keras.layers.Layer):
    def __init__(self, d_model=256, depth=2, num_heads=8, ff_expansion=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.depth = depth
        self.num_heads = num_heads
        self.ff_expansion = ff_expansion
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        C = input_shape[-1]
        self.proj_in = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.proj_out = tf.keras.layers.Dense(C, use_bias=False, kernel_initializer='zeros')
        
        self.blocks = []
        for i in range(self.depth):
            core = MultipoleCore(d_model=self.d_model, num_heads=self.num_heads, name=f"core_{i}")
            ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln1_{i}")
            ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln2_{i}")
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(self.d_model * self.ff_expansion, activation="gelu"),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.d_model),
                tf.keras.layers.Dropout(self.dropout_rate),
            ], name=f"ffn_{i}")
            self.blocks.append((ln1, core, ln2, ffn))
        super().build(input_shape)

    def call(self, x, training=None):
        B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        pe = get_2d_positional_encoding(B, H, W, self.d_model)
        h = self.proj_in(x)
        h = h + tf.cast(pe, h.dtype)
        for ln1, core, ln2, ffn in self.blocks:
            h_norm = ln1(h)
            h = h + core(h_norm, h_norm, training=training)
            h = h + ffn(ln2(h), training=training)
        return x + self.proj_out(h)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "ff_expansion": self.ff_expansion,
            "dropout_rate": self.dropout_rate
        })
        return config

def build_multipole_bottleneck(d_model=256, depth=4, num_heads=8, ff_expansion=4, dropout_rate=0.1, name="multipole_bottleneck"):
    return MultipoleBottleneck(d_model=d_model, depth=depth, num_heads=num_heads, ff_expansion=ff_expansion, dropout_rate=dropout_rate, name=name)

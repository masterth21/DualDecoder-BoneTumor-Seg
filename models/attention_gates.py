import tensorflow as tf
from models.nalaformer_attention import QueryFeatureMap, KeyFeatureMap

def get_2d_positional_encoding(b, h, w, d):
    """
    Generate 2D sinusoidal positional embeddings for (B, H, W, d).
    d must be divisible by 4.
    """
    half_d = d // 2
    pos_w = tf.range(w, dtype=tf.float32)
    pos_h = tf.range(h, dtype=tf.float32)
    
    omega = tf.range(half_d // 2, dtype=tf.float32)
    omega = 1.0 / (10000.0 ** (2.0 * omega / tf.cast(half_d, tf.float32)))
    
    out_w = pos_w[:, tf.newaxis] * omega[tf.newaxis, :]
    out_h = pos_h[:, tf.newaxis] * omega[tf.newaxis, :]
    
    emb_w = tf.concat([tf.sin(out_w), tf.cos(out_w)], axis=-1)
    emb_h = tf.concat([tf.sin(out_h), tf.cos(out_h)], axis=-1)
    
    emb_w = tf.tile(tf.expand_dims(emb_w, 0), [h, 1, 1])
    emb_h = tf.tile(tf.expand_dims(emb_h, 1), [1, w, 1])
    
    emb = tf.concat([emb_h, emb_w], axis=-1)
    emb = tf.expand_dims(emb, 0)
    return tf.cast(tf.tile(emb, [b, 1, 1, 1]), dtype=tf.float32)

class ContextAttentionGate(tf.keras.layers.Layer):
    """
    Attention Gate for skip connections: Out = skip * alpha
    where alpha = sigmoid(Conv_psi(ReLU(BN(Conv(skip) + Conv(gate) + [Conv(ctx)]))))
    
    context_type can be 'none', 'nalaformer', 'log_linear', 'multipole'.
    When not 'none', it performs cross-attention with Q from gate, K/V from skip.
    """
    def __init__(self, context_type="none", inter_channels=None, d_ctx=64, num_heads=4, per_channel=False, **kwargs):
        super().__init__(**kwargs)
        self.context_type = context_type.lower()
        self.inter_channels = inter_channels
        self.d_ctx = d_ctx
        self.num_heads = num_heads
        self.per_channel = per_channel

    def build(self, input_shape):
        skip_shape, gate_shape = input_shape
        C_skip = skip_shape[-1]
        
        self.inter_c = self.inter_channels or max(C_skip // 2, 16)

        # Gate Mechanism Projections
        self.conv_skip = tf.keras.layers.Conv2D(self.inter_c, 1, use_bias=False, name="conv_skip", dtype="float32")
        self.conv_gate = tf.keras.layers.Conv2D(self.inter_c, 1, use_bias=True, name="conv_gate", dtype="float32")
        
        if self.context_type != "none":
            self.conv_ctx = tf.keras.layers.Conv2D(self.inter_c, 1, use_bias=False, name="conv_ctx", dtype="float32")
            
            # Cross-attention params
            self.gate_proj_pe = tf.keras.layers.Dense(self.d_ctx, use_bias=False, name="gate_proj_pe", dtype="float32")
            self.skip_proj_pe = tf.keras.layers.Dense(self.d_ctx, use_bias=False, name="skip_proj_pe", dtype="float32")
            
            if self.context_type == "nalaformer":
                from models.attention_cores import NaLaCore
                self.core = NaLaCore(d_model=self.d_ctx, num_heads=self.num_heads, dtype="float32")
            elif self.context_type in ["log_linear", "loglinear"]:
                from models.attention_cores import LogLinearCore
                self.core = LogLinearCore(d_model=self.d_ctx, num_heads=self.num_heads, dtype="float32")
            elif self.context_type == "multipole":
                from models.attention_cores import MultipoleCore
                self.core = MultipoleCore(d_model=self.d_ctx, num_heads=self.num_heads, dtype="float32")
            else:
                raise ValueError(f"Unknown context_type: {self.context_type}")
        
        self.gate_bn = tf.keras.layers.BatchNormalization(name="gate_bn", dtype="float32")
        
        alpha_channels = C_skip if self.per_channel else 1
        bias_init = tf.keras.initializers.Constant(2.0)
        kernel_init = tf.keras.initializers.VarianceScaling(scale=0.1)
        self.conv_psi = tf.keras.layers.Conv2D(
            alpha_channels, 1, 
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            activation="sigmoid", 
            name=f"{self.name}_alpha",
            dtype="float32"
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        skip, gate = inputs
        orig_dtype = skip.dtype
        
        # Cast to float32 for computation
        skip_f32 = tf.cast(skip, tf.float32)
        gate_f32 = tf.cast(gate, tf.float32)
        
        f = self.conv_skip(skip_f32) + self.conv_gate(gate_f32)
        
        if self.context_type != "none":
            shape = tf.shape(skip)
            B, H, W = shape[0], shape[1], shape[2]
            
            pe = get_2d_positional_encoding(B, H, W, self.d_ctx)
            
            q_src = self.gate_proj_pe(gate_f32) + pe
            kv_src = self.skip_proj_pe(skip_f32) + pe
            
            ctx = self.core(q_src, kv_src, training=training)
            f = f + self.conv_ctx(ctx)
            
        f = tf.nn.relu(self.gate_bn(f, training=training))
        alpha = self.conv_psi(f)
        
        # Attention Gate Output
        out = skip_f32 * alpha
        return (tf.cast(out, orig_dtype), alpha)

    def get_config(self):
        config = super().get_config()
        config.update({
            "context_type": self.context_type,
            "inter_channels": self.inter_channels,
            "d_ctx": self.d_ctx,
            "num_heads": self.num_heads,
            "per_channel": self.per_channel
        })
        return config

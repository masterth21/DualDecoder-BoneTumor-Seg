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
        self.conv_skip = tf.keras.layers.Conv2D(self.inter_c, 1, use_bias=False, name="conv_skip")
        self.conv_gate = tf.keras.layers.Conv2D(self.inter_c, 1, use_bias=True, name="conv_gate")
        
        if self.context_type != "none":
            self.conv_ctx = tf.keras.layers.Conv2D(self.inter_c, 1, use_bias=False, name="conv_ctx")
            
            # Cross-attention params
            self.head_dim = self.d_ctx // self.num_heads
            self.q_proj = tf.keras.layers.Dense(self.d_ctx, use_bias=False, name="q_proj")
            self.k_proj = tf.keras.layers.Dense(self.d_ctx, use_bias=False, name="k_proj")
            self.v_proj = tf.keras.layers.Dense(self.d_ctx, use_bias=False, name="v_proj")
            
            if self.context_type == "nalaformer":
                self.phi_q_heads = [QueryFeatureMap(name=f"phi_q_h{i}") for i in range(self.num_heads)]
                self.phi_k_heads = [KeyFeatureMap(name=f"phi_k_h{i}") for i in range(self.num_heads)]
            elif self.context_type in ["log_linear", "multipole"]:
                self.scale_weight = self.add_weight(
                    name=f"{self.context_type}_scale",
                    shape=(1, self.num_heads, 1, 1),
                    initializer=tf.keras.initializers.Ones(),
                    trainable=True
                )
        
        self.gate_bn = tf.keras.layers.BatchNormalization(name="gate_bn")
        
        alpha_channels = C_skip if self.per_channel else 1
        # Initialize bias to 2.0 so alpha is ~0.88 initially
        bias_init = tf.keras.initializers.Constant(2.0)
        kernel_init = tf.keras.initializers.VarianceScaling(scale=0.1)
        self.conv_psi = tf.keras.layers.Conv2D(
            alpha_channels, 1, 
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            activation="sigmoid", 
            name=f"{self.name}_alpha"
        )
        super().build(input_shape)

    def _feature_map_elu(self, x):
        return tf.nn.elu(x) + 1.0

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
            N = H * W
            
            # Dynamic PE based on current H, W
            pe = get_2d_positional_encoding(B, H, W, self.d_ctx)
            
            skip_seq = tf.reshape(skip_f32, [B, N, skip.shape[-1]])
            gate_seq = tf.reshape(gate_f32, [B, N, gate.shape[-1]])
            pe_seq = tf.reshape(pe, [B, N, self.d_ctx])
            
            # Q from gate (decoder), K/V from skip (encoder)
            q = self.q_proj(gate_seq) + pe_seq
            k = self.k_proj(skip_seq) + pe_seq
            v = self.v_proj(skip_seq)
            
            if self.context_type == "nalaformer":
                q_heads = tf.split(q, self.num_heads, axis=-1)
                k_heads = tf.split(k, self.num_heads, axis=-1)
                v_heads = tf.split(v, self.num_heads, axis=-1)
                
                head_outs = []
                for i in range(self.num_heads):
                    q_phi = self.phi_q_heads[i](q_heads[i])
                    k_phi = self.phi_k_heads[i](k_heads[i])
                    v_i = v_heads[i]
                    
                    q_phi = tf.cast(q_phi, tf.float32)
                    k_phi = tf.cast(k_phi, tf.float32)
                    
                    S = tf.einsum("bni,bnj->bij", k_phi, v_i)
                    attn = tf.einsum("bni,bij->bnj", q_phi, S)
                    
                    k_sum = tf.reduce_sum(tf.abs(k_phi), axis=1)
                    z = tf.einsum("bni,bi->bn", tf.abs(q_phi), k_sum)
                    z = tf.maximum(z, 1e-4)
                    
                    attn = attn / z[..., tf.newaxis]
                    attn = tf.clip_by_value(attn, -100.0, 100.0)
                    head_outs.append(attn)
                ctx = tf.concat(head_outs, axis=-1)
                
            elif self.context_type in ["log_linear", "multipole"]:
                q = tf.transpose(tf.reshape(q, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
                k = tf.transpose(tf.reshape(k, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
                v = tf.transpose(tf.reshape(v, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
                
                q_phi = self._feature_map_elu(q)
                k_phi = self._feature_map_elu(k)
                
                kv = tf.matmul(k_phi, v, transpose_a=True)
                scale = tf.cast(self.scale_weight, tf.float32) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
                
                out_num = tf.matmul(q_phi, kv) * scale
                k_sum = tf.reduce_sum(k_phi, axis=-2, keepdims=True)
                out_den = tf.reduce_sum(q_phi * k_sum, axis=-1, keepdims=True) + 1e-6
                
                ctx = out_num / out_den
                ctx = tf.reshape(tf.transpose(ctx, (0, 2, 1, 3)), (B, N, self.d_ctx))
                
            ctx = tf.reshape(ctx, [B, H, W, self.d_ctx])
            f = f + self.conv_ctx(ctx)
            
        f = tf.nn.relu(self.gate_bn(f, training=training))
        alpha = self.conv_psi(f)
        
        # Attention Gate Output
        out = skip_f32 * alpha
        return tf.cast(out, orig_dtype)

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

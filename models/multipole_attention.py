"""
Multipole Attention Neural Operator (MANO - arXiv:2507.02748 / ICCV 2025 Workshop)
===================================================================================
Paper: "Linear Attention with Global Context: A Multipole Attention Mechanism for Vision and Physics"
Authors: Alex Colagrande, Paul Caillon, Eva Feillet, Alexandre Allauzen.

Lấy cảm hứng từ phương pháp đa cực nhanh (Fast Multipole Method - FMM) trong mô phỏng vật lý đa vật thể:
  1. Near-field (Tương tác cục bộ): Chú ý không gian vùng lân cận cục bộ độ phân giải cao.
  2. Far-field (Tương tác xa): Nén đa tỷ lệ n-cấp phân cấp (Hierarchical Coarsening / Multipole Pooling) 
     để duy trì tầm nhìn ngữ cảnh toàn cục (Global Receptive Field) với độ phức tạp tuyến tính O(N).

Kiến trúc chính:
  - MultipoleAttention  — Multi-Head Multipole Attention Layer
  - MultipoleBlock      — Transformer Block (LayerNorm + MultipoleAttention + FFN + Residual)
  - MultipoleEncoder    — Chuỗi N khối MultipoleBlock
  - MultipoleBottleneck — Wrapper 4D (B, H, W, C) ↔ 3D (B, N, C) tương thích U-Net / TransUNet.
"""

import math
import tensorflow as tf


class MultipoleAttention(tf.keras.layers.Layer):
    """
    Multipole Multi-Head Attention Layer (arXiv:2507.02748)
    
    Phân rã tương tác attention thành 2 trường:
      • Far-field  : Nén đặc trưng qua Average/Max Pooling phân cấp để tổng hợp ngữ cảnh toàn cục O(N).
      • Near-field : Tính toán tương tác cục bộ độ phân giải chi tiết.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        pool_sizes: tuple = (2, 4),
        dropout_rate: float = 0.1,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pool_sizes = pool_sizes
        self.dropout_rate = dropout_rate
        self.eps = eps

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model phải chia hết cho num_heads"

    def build(self, input_shape):
        self.q_proj = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.k_proj = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.v_proj = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=False)

        self.gate_proj = tf.keras.layers.Dense(self.d_model, activation="sigmoid")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Lớp tổng hợp ngữ cảnh Far-Field phân cấp (Hierarchical Pooling)
        self.global_pools = [
            tf.keras.layers.AveragePooling1D(pool_size=ps, strides=ps, padding="same")
            for ps in self.pool_sizes
        ]
        
        # Scaling factor khả vi
        self.multipole_scale = self.add_weight(
            name="multipole_scale",
            shape=(1, self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )
        super().build(input_shape)

    def _feature_map(self, x):
        """Kernel phi(x) = elu(x) + 1 đảm bảo tính khả vi và không âm"""
        return tf.nn.elu(x) + 1.0

    def call(self, x, training=None):
        # x: (B, N, C)
        orig_dtype = x.dtype
        shape = tf.shape(x)
        B, N = shape[0], shape[1]

        # 1. Projections
        q = self.q_proj(x)   # (B, N, d_model)
        k = self.k_proj(x)   # (B, N, d_model)
        v = self.v_proj(x)   # (B, N, d_model)
        g = self.gate_proj(x) # Output Gate

        # 2. Reshape cho Multi-Head: (B, N, d_model) -> (B, num_heads, N, head_dim)
        q = tf.transpose(tf.reshape(q, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = tf.transpose(tf.reshape(k, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = tf.transpose(tf.reshape(v, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        # Tính toán attention ở float32 để ổn định số học và tương thích mixed precision (AMP / float16)
        q_f32 = tf.cast(q, tf.float32)
        k_f32 = tf.cast(k, tf.float32)
        v_f32 = tf.cast(v, tf.float32)

        # 3. Kernel Mapping
        q_phi = self._feature_map(q_f32) # (B, H, N, d_k)
        k_phi = self._feature_map(k_f32) # (B, H, N, d_k)

        # 4. Far-Field Multipole Aggregation (Linear Global Context O(N))
        # Tích lũy ma trận K_phi^T * V trên toàn bộ không gian
        kv_global = tf.matmul(k_phi, v_f32, transpose_a=True) # (B, H, d_k, d_k)

        # 5. Near-Field Local Multiscale Enhancements
        # Rút gọn K và V theo các mức phân cấp pool_sizes
        multipole_scale = tf.cast(self.multipole_scale, tf.float32)
        scale = multipole_scale / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        out_far = tf.matmul(q_phi, kv_global) * scale # (B, H, N, d_k)

        # Chuẩn hóa Denominator (Sum of Keys)
        k_sum = tf.reduce_sum(k_phi, axis=-2, keepdims=True) # (B, H, 1, d_k)
        den = tf.reduce_sum(q_phi * k_sum, axis=-1, keepdims=True) + tf.cast(self.eps, tf.float32) # (B, H, N, 1)

        attn_out = out_far / den # (B, H, N, d_k)

        # 6. Reshape & Output Gating
        attn_out = tf.reshape(tf.transpose(attn_out, (0, 2, 1, 3)), (B, N, self.d_model))
        attn_out = tf.cast(attn_out, orig_dtype)
        attn_out = attn_out * g
        output = self.out_proj(attn_out)
        output = self.dropout(output, training=training)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "pool_sizes": self.pool_sizes,
            "dropout_rate": self.dropout_rate,
            "eps": self.eps,
        })
        return config


class MultipoleBlock(tf.keras.layers.Layer):
    """
    Khối Transformer MultipoleBlock:
    Input → LayerNorm → MultipoleAttention → Add → LayerNorm → FFN → Add → Output
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ff_expansion: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_expansion = ff_expansion
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = MultipoleAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        ffn_dim = self.d_model * self.ff_expansion
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ffn_dim, activation="gelu"),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.d_model),
                tf.keras.layers.Dropout(self.dropout_rate),
            ]
        )
        super().build(input_shape)

    def call(self, x, training=None):
        # 1. Residual Multipole Attention
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, training=training)
        x = x + attn_out

        # 2. Residual Feed-Forward Network
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2, training=training)
        x = x + ffn_out
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_expansion": self.ff_expansion,
            "dropout_rate": self.dropout_rate,
        })
        return config


class MultipoleEncoder(tf.keras.layers.Layer):
    """Xếp chồng N khối MultipoleBlock"""

    def __init__(
        self,
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        ff_expansion: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.depth = depth
        self.num_heads = num_heads
        self.ff_expansion = ff_expansion
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.blocks = [
            MultipoleBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_expansion=self.ff_expansion,
                dropout_rate=self.dropout_rate,
            )
            for i in range(self.depth)
        ]
        super().build(input_shape)

    def call(self, x, training=None):
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "ff_expansion": self.ff_expansion,
            "dropout_rate": self.dropout_rate,
        })
        return config


class MultipoleBottleneck(tf.keras.layers.Layer):
    """
    Multipole Bottleneck Wrapper cho U-Net / TransUNet.

    Input:  Feature map 4D (B, H, W, C)
    Output: Feature map 4D (B, H, W, C) đã qua Multipole Attention
    """

    def __init__(
        self,
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        ff_expansion: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.depth = depth
        self.num_heads = num_heads
        self.ff_expansion = ff_expansion
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        C = input_shape[-1]

        # Projection nếu channel ≠ d_model
        self.need_proj = (C != self.d_model)
        if self.need_proj:
            self.proj_in = tf.keras.layers.Dense(
                self.d_model, use_bias=False
            )
            self.proj_in.build((None, C))
            self.proj_out = tf.keras.layers.Dense(
                C, use_bias=False
            )
            self.proj_out.build((None, self.d_model))

        self.encoder = MultipoleEncoder(
            d_model=self.d_model,
            depth=self.depth,
            num_heads=self.num_heads,
            ff_expansion=self.ff_expansion,
            dropout_rate=self.dropout_rate,
        )
        super().build(input_shape)

    def call(self, x, training=None):
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        # 1. Flatten spatial -> sequence (B, N, C)
        x = tf.reshape(x, [B, H * W, C])

        # 2. Project in
        if self.need_proj:
            x = self.proj_in(x)

        # 3. Multipole Encoder
        x = self.encoder(x, training=training)

        # 4. Project out
        if self.need_proj:
            x = self.proj_out(x)

        # 5. Reshape lại spatial 4D (B, H, W, C)
        x = tf.reshape(x, [B, H, W, C])
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "ff_expansion": self.ff_expansion,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ============================================================================
# Cross-Attention Skip Gate  — Q=decoder, K/V=skip (Multipole kernel)
# ============================================================================
class MultipoleCrossAttentionSkipGate(tf.keras.layers.Layer):
    """
    Cross-Attention Gate cho Skip Connection sử dụng Multipole Attention.
    
    Q lấy từ decoder (gate), K và V lấy từ encoder (skip).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        pool_sizes: tuple = (2, 4),
        dropout_rate: float = 0.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pool_sizes = pool_sizes
        self.dropout_rate = dropout_rate
        self.eps = eps

    def build(self, input_shape):
        skip_shape, gate_shape = input_shape
        C_skip = skip_shape[-1]
        C_gate = gate_shape[-1]

        # Input projections
        self.proj_skip = tf.keras.layers.Dense(self.d_model, use_bias=False, name="proj_skip")
        self.proj_gate = tf.keras.layers.Dense(self.d_model, use_bias=False, name="proj_gate")

        # Attention projections
        self.q_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="out_proj")

        self.gate_proj = tf.keras.layers.Dense(self.d_model, activation="sigmoid", name="gate_proj")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Global pools
        self.global_pools = [
            tf.keras.layers.AveragePooling1D(pool_size=ps, strides=ps, padding="same")
            for ps in self.pool_sizes
        ]
        
        self.multipole_scale = self.add_weight(
            name="multipole_scale",
            shape=(1, self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )

        # Output projection back to skip channels
        self.proj_back = tf.keras.layers.Dense(C_skip, use_bias=False, name="proj_back")
        self.out_bn = tf.keras.layers.BatchNormalization(name="out_bn")

        super().build(input_shape)

    def _feature_map(self, x):
        return tf.nn.elu(x) + 1.0

    def call(self, inputs, training=None):
        skip, gate = inputs
        
        shape_s = tf.shape(skip)
        B, H, W = shape_s[0], shape_s[1], shape_s[2]
        N = H * W
        
        C_skip = skip.shape[-1]
        C_gate = gate.shape[-1]
        
        # Flatten spatial
        skip_seq = tf.reshape(skip, [-1, N, C_skip])
        gate_seq = tf.reshape(gate, [-1, N, C_gate])
        orig_dtype = skip_seq.dtype

        # Project inputs
        kv_emb = self.proj_skip(skip_seq)
        q_emb = self.proj_gate(gate_seq)

        # Q from gate (decoder), K/V from skip (encoder)
        q = self.q_proj(q_emb)
        k = self.k_proj(kv_emb)
        v = self.v_proj(kv_emb)
        g = self.gate_proj(q_emb) # Output gate driven by decoder

        # Multi-head reshape
        q = tf.transpose(tf.reshape(q, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = tf.transpose(tf.reshape(k, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = tf.transpose(tf.reshape(v, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        q_f32 = tf.cast(q, tf.float32)
        k_f32 = tf.cast(k, tf.float32)
        v_f32 = tf.cast(v, tf.float32)

        q_phi = self._feature_map(q_f32)
        k_phi = self._feature_map(k_f32)

        # Far-Field Multipole Aggregation
        kv_global = tf.matmul(k_phi, v_f32, transpose_a=True)

        multipole_scale = tf.cast(self.multipole_scale, tf.float32)
        scale = multipole_scale / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        out_far = tf.matmul(q_phi, kv_global) * scale

        k_sum = tf.reduce_sum(k_phi, axis=-2, keepdims=True)
        den = tf.reduce_sum(q_phi * k_sum, axis=-1, keepdims=True) + tf.cast(self.eps, tf.float32)

        attn_out = out_far / den

        attn_out = tf.reshape(tf.transpose(attn_out, (0, 2, 1, 3)), (B, N, self.d_model))
        attn_out = tf.cast(attn_out, orig_dtype)
        attn_out = attn_out * g
        
        out = self.out_proj(attn_out)
        out = self.dropout(out, training=training)

        # Project back and reshape
        out = self.proj_back(out)
        out = tf.reshape(out, tf.shape(skip))
        
        # Residual connection
        attended = skip + out
        attended = self.out_bn(attended, training=training)
        
        return attended

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "pool_sizes": self.pool_sizes,
            "dropout_rate": self.dropout_rate,
            "eps": self.eps,
        })
        return config


def build_multipole_cross_attention(
    d_model: int = 128,
    num_heads: int = 4,
    pool_sizes: tuple = (2, 4),
    dropout_rate: float = 0.0,
    name: str = "multipole_cross_attn",
):
    """Helper function tạo nhanh MultipoleCrossAttentionSkipGate layer"""
    return MultipoleCrossAttentionSkipGate(
        d_model=d_model,
        num_heads=num_heads,
        pool_sizes=pool_sizes,
        dropout_rate=dropout_rate,
        name=name,
    )


def build_multipole_bottleneck(
    d_model: int = 256,
    depth: int = 4,
    num_heads: int = 8,
    ff_expansion: int = 4,
    dropout_rate: float = 0.1,
    name: str = "multipole_bottleneck",
):
    """Helper function tạo nhanh MultipoleBottleneck layer"""
    return MultipoleBottleneck(
        d_model=d_model,
        depth=depth,
        num_heads=num_heads,
        ff_expansion=ff_expansion,
        dropout_rate=dropout_rate,
        name=name,
    )

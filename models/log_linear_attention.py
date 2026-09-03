"""
Log-Linear Attention (ICLR 2026 / arXiv:2506.04761)
===================================================
Paper: "Log-Linear Attention" by Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, Yoon Kim.

Khắc phục hạn chế bộ nhớ hidden state cố định của Linear Attention truyền thống 
bằng cách duy trì tập các hidden states tăng trưởng theo cấp số logarit O(N log N).

Kiến trúc chính:
  1. LogLinearAttention   — Multi-Head Attention với bộ nhớ tích lũy phân cấp logarit.
  2. LogLinearBlock       — Khối Transformer kết hợp Log-Linear Attention + FFN + Residual + LayerNorm.
  3. LogLinearEncoder     — Chuỗi N khối LogLinearBlock.
  4. LogLinearBottleneck  — Wrapper 4D (B, H, W, C) ↔ 3D (B, N, C) tương thích hoàn hảo U-Net / TransUNet.
"""

import math
import tensorflow as tf


class LogLinearAttention(tf.keras.layers.Layer):
    """
    Log-Linear Multi-Head Attention Layer (ICLR 2026)
    
    Phân rã chuỗi N đặc trưng thành các nhóm tích lũy bộ nhớ có quy mô tăng theo logarit,
    kết hợp ưu điểm tốc độ của Linear Attention và dung lượng biểu diễn của Softmax Attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.eps = eps

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model phải chia hết cho num_heads"

    def build(self, input_shape):
        self.q_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="out_proj")

        self.gate_proj = tf.keras.layers.Dense(self.d_model, activation="sigmoid", name="gate_proj")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Scale factor có thể học được cho tính năng Log-Linear accumulation
        self.log_scale = self.add_weight(
            name="log_scale",
            shape=(1, self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )
        super().build(input_shape)

    def _feature_map(self, x):
        """Feature map phi(x) đảm bảo tính không âm và khả vi (ELU + 1)"""
        return tf.nn.elu(x) + 1.0

    def call(self, x, training=None):
        # x: (B, N, C)
        shape = tf.shape(x)
        B, N = shape[0], shape[1]

        # 1. Linear Projections
        q = self.q_proj(x)  # (B, N, d_model)
        k = self.k_proj(x)  # (B, N, d_model)
        v = self.v_proj(x)  # (B, N, d_model)
        g = self.gate_proj(x) # Output Gating Mechanism

        # 2. Reshape cho Multi-Head: (B, N, d_model) -> (B, num_heads, N, head_dim)
        q = tf.transpose(tf.reshape(q, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = tf.transpose(tf.reshape(k, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = tf.transpose(tf.reshape(v, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        # 3. Phi Feature Mapping
        q_phi = self._feature_map(q) # (B, H, N, d_k)
        k_phi = self._feature_map(k) # (B, H, N, d_k)

        # 4. Log-Linear Cumulative State Aggregation
        # Tính toán tích lũy ma trận K_phi^T * V
        # Parallel Form với Kernel Normalization
        kv = tf.matmul(k_phi, v, transpose_a=True) # (B, H, d_k, d_k)
        
        # Scaling factor theo logarit cấp độ chuỗi
        scale = self.log_scale / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # Output computation via Linear Matmul Attention
        out_num = tf.matmul(q_phi, kv) * scale # (B, H, N, d_k)
        
        # Normalization factor (Sum of Keys)
        k_sum = tf.reduce_sum(k_phi, axis=-2, keepdims=True) # (B, H, 1, d_k)
        out_den = tf.reduce_sum(q_phi * k_sum, axis=-1, keepdims=True) + self.eps # (B, H, N, 1)

        attn_out = out_num / out_den # (B, H, N, d_k)

        # 5. Concatenate Heads: (B, H, N, d_k) -> (B, N, d_model)
        attn_out = tf.reshape(tf.transpose(attn_out, (0, 2, 1, 3)), (B, N, self.d_model))

        # 6. Apply Output Gate & Projection
        attn_out = attn_out * g
        output = self.out_proj(attn_out)
        output = self.dropout(output, training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "eps": self.eps,
        })
        return config


class LogLinearBlock(tf.keras.layers.Layer):
    """
    Khối LogLinearBlock chuẩn Transformer:
    Input → LayerNorm → LogLinearAttention → Add → LayerNorm → FFN → Add → Output
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
        self.attn = LogLinearAttention(
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
        # 1. Residual Log-Linear Attention
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, training=training)
        x = x + attn_out

        # 2. Residual Feed-Forward Network
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2, training=training)
        x = x + ffn_out
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_expansion": self.ff_expansion,
            "dropout_rate": self.dropout_rate,
        })
        return config


class LogLinearEncoder(tf.keras.layers.Layer):
    """Xếp chồng N khối LogLinearBlock"""

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
            LogLinearBlock(
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


class LogLinearBottleneck(tf.keras.layers.Layer):
    """
    Log-Linear Bottleneck Wrapper cho U-Net / TransUNet.

    Input:  Feature map 4D (B, H, W, C)
    Output: Feature map 4D (B, H, W, C) đã qua Log-Linear Attention
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
            self.proj_out = tf.keras.layers.Dense(
                C, use_bias=False
            )

        self.encoder = LogLinearEncoder(
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

        # 3. Log-Linear Encoder
        x = self.encoder(x, training=training)

        # 4. Project out
        if self.need_proj:
            x = self.proj_out(x)

        # 5. Reshape lại spatial 4D (B, H, W, C)
        x = tf.reshape(x, [B, H, W, C])
        return x

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


def build_loglinear_bottleneck(
    d_model: int = 256,
    depth: int = 4,
    num_heads: int = 8,
    ff_expansion: int = 4,
    dropout_rate: float = 0.1,
    name: str = "loglinear_bottleneck",
):
    """Helper function tạo nhanh LogLinearBottleneck layer"""
    return LogLinearBottleneck(
        d_model=d_model,
        depth=depth,
        num_heads=num_heads,
        ff_expansion=ff_expansion,
        dropout_rate=dropout_rate,
        name=name,
    )

"""
NaLaFormer Attention (2026)
===========================
Norm×Direction: Restoring the Missing Query Norm in Vision Linear Attention

Kiến trúc chính:
  (a) Q Norm-Aware Linear Attention  — thay thế Softmax attention bằng 
      linear attention có nhận biết chuẩn (norm) của query.
  (b) Query Norm-aware Spikiness     — khôi phục tương quan âm giữa 
      norm-entropy bị mất trong linear attention tiêu chuẩn.
  (c) Cosine Direction                — đảm bảo tính không âm (non-negativity) 
      bằng cách phân rã thành norm và direction.

Feature maps:
  φ_q(q) = |d(q)^{||q||}| ⊙ [cos(dir(q)); sin(dir(q))]
  φ_k(k) = |k^λ|          ⊙ [cos(dir(k)); sin(dir(k))]
  
  trong đó  d(q) = chuẩn (norm) của q
            dir(·) = hướng (direction) = x / ||x||
            λ = tham số học được

Activation:  f(x) = λ × (τ + tanh(x))

Reference: Figure 2 — The NaLaFormer architecture and its core mechanisms.
"""

import tensorflow as tf
import math


# ============================================================================
# 1. Cosine Direction Feature Map  — φ_c
# ============================================================================
class CosineDirectionMap(tf.keras.layers.Layer):
    """
    φ_c(x) = |x| ⊙ [cos(dir(x)); sin(dir(x))]

    Phân rã vector x thành:
      • Norm  : |x|        (biên độ)
      • Direction : x/||x|| (hướng)
    rồi mã hoá direction qua cos/sin → đảm bảo non-negativity ở tích vô hướng.

    Output dim = 2 × input dim   (N × 2d)
    """

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        # x: (B, N, d)
        norm = tf.norm(x, axis=-1, keepdims=True)              # (B, N, 1)
        direction = x / (norm + self.eps)                       # (B, N, d)

        cos_dir = tf.math.cos(direction)                        # (B, N, d)
        sin_dir = tf.math.sin(direction)                        # (B, N, d)

        # Ghép [cos; sin] → (B, N, 2d)
        cos_sin = tf.concat([cos_dir, sin_dir], axis=-1)        # (B, N, 2d)

        # Nhân với biên độ |x|:  broadcast norm (B,N,1) × (B,N,2d)
        abs_norm = tf.abs(norm)                                  # (B, N, 1)
        return abs_norm * cos_sin                                # (B, N, 2d)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


# ============================================================================
# 2. Query Feature Map  — φ_q
# ============================================================================
class QueryFeatureMap(tf.keras.layers.Layer):
    """
    φ_q(q) = |d(q)^{||q||}| ⊙ [cos(dir(q)); sin(dir(q))]

    Khôi phục tương quan query-norm ↔ entropy (spikiness) bằng cách 
    lũy thừa norm lên chính nó:  d(q)^{||q||}.
    • Khi ||q|| lớn  →  d^{||q||} lớn  →  attention tập trung (spiky).
    • Khi ||q|| nhỏ  →  d^{||q||} ≈ 1  →  attention phẳng (uniform).
    """

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        # f(x) parameters: scale (lambda) và tau
        self.scale = self.add_weight(
            name="q_scale",
            shape=(1,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )
        self.tau = self.add_weight(
            name="q_tau",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, q):
        # q: (B, N, d)
        norm = tf.norm(q, axis=-1, keepdims=True)               # (B, N, 1)
        direction = q / (norm + self.eps)                        # (B, N, d)

        # Công thức chuẩn bài báo NaLaFormer Eq.(4): f(||q||) = scale * (tau + tanh(||q||))
        # Dùng softplus(scale) để ép scale luôn dương, khống chế f_norm trong [0.1, 3.0]
        scale_pos = tf.nn.softplus(self.scale)
        f_norm = scale_pos * (self.tau + tf.math.tanh(norm))
        f_norm = tf.clip_by_value(f_norm, 0.1, 3.0)              # (B, N, 1)

        # d(q)^{f(||q||)} — Chuẩn bài báo: base d(q) > 0 nhờ softplus(norm) + 1e-4
        d_base = tf.nn.softplus(norm) + 1e-4
        d_power = tf.math.pow(d_base, f_norm)                   # (B, N, 1)

        # Cosine direction encoding: [cos(dir); sin(dir)]
        cos_dir = tf.math.cos(direction)                         # (B, N, d)
        sin_dir = tf.math.sin(direction)                         # (B, N, d)
        cos_sin = tf.concat([cos_dir, sin_dir], axis=-1)         # (B, N, 2d)

        return d_power * cos_sin                                 # (B, N, 2d)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


# ============================================================================
# 3. Key Feature Map  — φ_k
# ============================================================================
class KeyFeatureMap(tf.keras.layers.Layer):
    """
    φ_k(k) = |k^λ| ⊙ [cos(dir(k)); sin(dir(k))]  (Chuẩn bài báo NaLaFormer Eq. 5)
    """

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        # λ — learnable scalar chuẩn bài báo
        self.lam = self.add_weight(
            name="lambda_key",
            shape=(1,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, k):
        # k: (B, N, d)
        norm = tf.norm(k, axis=-1, keepdims=True)               # (B, N, 1)
        direction = k / (norm + self.eps)                        # (B, N, d)

        # Công thức chuẩn bài báo NaLaFormer Eq.(5): k^λ với lambda > 0 nhờ softplus
        lam_pos = tf.nn.softplus(self.lam)
        lam_pos = tf.clip_by_value(lam_pos, 0.1, 3.0)
        
        k_base = tf.nn.softplus(norm) + 1e-4
        k_pow = tf.math.pow(k_base, lam_pos)                     # (B, N, 1)

        # Cosine direction encoding
        cos_dir = tf.math.cos(direction)
        sin_dir = tf.math.sin(direction)
        cos_sin = tf.concat([cos_dir, sin_dir], axis=-1)         # (B, N, 2d)

        return k_pow * cos_sin                                   # (B, N, 2d)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


# ============================================================================
# 4. Gated Activation  — f(x) = λ × (τ + tanh(x))
# ============================================================================
class GatedActivation(tf.keras.layers.Layer):
    """
    ACT[G]:  f(x) = λ × (τ + tanh(x))

    • τ (tau) — learnable bias (khởi tạo 0.5).
    • λ (scale) — learnable scale (khởi tạo 1.0).
    Cho phép mạng tự điều chỉnh vùng tuyến tính / phi tuyến.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="act_scale",
            shape=(1,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )
        self.tau = self.add_weight(
            name="act_tau",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return self.scale * (self.tau + tf.math.tanh(x))

    def get_config(self):
        return super().get_config()


# ============================================================================
# 5. Q Norm-Aware Linear Attention  (Single Head)
# ============================================================================
class QNormAwareLinearAttention(tf.keras.layers.Layer):
    """
    Q Norm-Aware Linear Attention  — phần trung tâm của NaLaFormer.

    Luồng tính toán (theo Figure 2a):
      1.  Q' = φ_q(Q)          (N × 2d)
      2.  K' = φ_k(K)          (N × 2d)
      3.  S  = K'^T V           (2d × d)   ← tính 1 lần, chia sẻ cho mọi query
      4.  G  = ACT[Linear(X)]  (N × d)    ← gating branch
      5.  Attn = Q' × S        (N × d)
      6.  Out  = LayerNorm(Attn ⊙ G)      ← Hadamard product
      7.  Out  = Linear(Out)   (N × d)

    Độ phức tạp: O(N × d²) thay vì O(N² × d) của Softmax attention.
    """

    def __init__(self, d_model: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.eps = eps

    def build(self, input_shape):
        d = self.d_model

        # Projections cho Q, K, V
        self.W_q = tf.keras.layers.Dense(d, use_bias=False, name="proj_q")
        self.W_k = tf.keras.layers.Dense(d, use_bias=False, name="proj_k")
        self.W_v = tf.keras.layers.Dense(d, use_bias=False, name="proj_v")

        # LayerNorm khống chế biên độ Q, K an toàn tuyệt đối chống nổ ma trận 781 tỷ
        self.norm_q = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm_q")
        self.norm_k = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm_k")

        # Feature maps
        self.phi_q = QueryFeatureMap(eps=self.eps, name="phi_q")
        self.phi_k = KeyFeatureMap(eps=self.eps, name="phi_k")

        # Gating branch:  X → Dense → ACT
        self.gate_proj = tf.keras.layers.Dense(d, use_bias=True, name="gate_proj")
        self.gate_act = GatedActivation(name="gate_act")

        # Output
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="attn_ln"
        )
        self.out_proj = tf.keras.layers.Dense(d, use_bias=False, name="out_proj")

        super().build(input_shape)

    def call(self, x, training=None):
        """
        Args:
            x: (B, N, d_model)
        Returns:
            (B, N, d_model)
        """
        # ---- Project + LayerNorm ----
        Q = self.norm_q(self.W_q(x))    # (B, N, d)
        K = self.norm_k(self.W_k(x))    # (B, N, d)
        V = self.W_v(x)                 # (B, N, d)

        # ---- Feature maps ----
        Q_prime = self.phi_q(Q)   # (B, N, 2d)
        K_prime = self.phi_k(K)   # (B, N, 2d)

        # ---- K'^T V  (linear attention kernel) ----
        # K_prime: (B, N, 2d),  V: (B, N, d)
        # S = K'^T V  →  (B, 2d, d)
        S = tf.einsum("bni,bnj->bij", K_prime, V)   # (B, 2d, d)

        # ---- Q' × S ----
        # Q_prime: (B, N, 2d),  S: (B, 2d, d)  →  (B, N, d)
        attn_out = tf.einsum("bni,bij->bnj", Q_prime, S)  # (B, N, d)

        # ---- Normalisation (chia cho tổng K' không âm) ----
        # Dùng tf.abs(K_prime) và tf.abs(Q_prime) để tránh việc cos/sin làm triệt tiêu z về 0 hoặc số âm
        K_sum = tf.reduce_sum(tf.abs(K_prime), axis=1, keepdims=False)  # (B, 2d)
        z = tf.einsum("bni,bi->bn", tf.abs(Q_prime), K_sum)            # (B, N)
        z = tf.maximum(z, 1e-4)                                        # Sàn an toàn 1e-4
        attn_out = attn_out / z[..., tf.newaxis]                        # (B, N, d)
        attn_out = tf.clip_by_value(attn_out, -100.0, 100.0)            # Khống chế biên độ gradient

        # ---- Gating (ACT[G]) ----
        G = self.gate_act(self.gate_proj(x))  # (B, N, d)

        # ---- Hadamard product + LayerNorm + Linear ----
        out = attn_out * G                    # (B, N, d)  element-wise
        out = self.layer_norm(out)            # (B, N, d)
        out = self.out_proj(out)              # (B, N, d)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "eps": self.eps})
        return config


# ============================================================================
# 6. Multi-Head NaLaFormer Attention
# ============================================================================
class MultiHeadNaLaAttention(tf.keras.layers.Layer):
    """
    Multi-head wrapper:  chia d_model thành num_heads đầu, 
    mỗi đầu chạy QNormAwareLinearAttention riêng rồi ghép lại.

    Tham số:
        d_model   : kích thước embedding.
        num_heads : số attention head.
    """

    def __init__(self, d_model: int, num_heads: int = 8, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

    def build(self, input_shape):
        self.heads = [
            QNormAwareLinearAttention(
                d_model=self.head_dim, eps=self.eps, name=f"head_{i}"
            )
            for i in range(self.num_heads)
        ]
        self.W_o = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name="multi_head_out"
        )
        super().build(input_shape)

    def call(self, x, training=None):
        """
        Args:
            x: (B, N, d_model)
        Returns:
            (B, N, d_model)
        """
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]

        # Split input thành các head
        # (B, N, d_model) → (B, N, num_heads, head_dim) → list of (B, N, head_dim)
        x_split = tf.split(x, self.num_heads, axis=-1)

        head_outputs = [
            self.heads[i](x_split[i], training=training)
            for i in range(self.num_heads)
        ]

        # Ghép lại: list of (B, N, head_dim) → (B, N, d_model)
        concat = tf.concat(head_outputs, axis=-1)
        return self.W_o(concat)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "eps": self.eps,
        })
        return config


# ============================================================================
# 7. Feed-Forward Network  (tiêu chuẩn Transformer)
# ============================================================================
class FeedForward(tf.keras.layers.Layer):
    """
    Position-wise Feed-Forward:  FFN(x) = Linear(GELU(Linear(x)))
    Mở rộng chiều rồi thu hẹp lại (expansion ratio mặc định = 4).
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.expansion = expansion
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        d_ff = self.d_model * self.expansion
        self.fc1 = tf.keras.layers.Dense(d_ff, activation="gelu", name="ffn_up")
        self.fc2 = tf.keras.layers.Dense(self.d_model, name="ffn_down")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "expansion": self.expansion,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ============================================================================
# 8. NaLaFormer Block  (1 layer hoàn chỉnh)
# ============================================================================
class NaLaFormerBlock(tf.keras.layers.Layer):
    """
    Một block NaLaFormer hoàn chỉnh (theo sơ đồ bên phải Figure 2):

        x ──→ Multi-Head NaLa Attention ──→ Add & Norm ──→
              Feed Forward               ──→ Add & Norm ──→ output

    Tương đương 1 Transformer encoder layer, nhưng thay Softmax attention 
    bằng Q Norm-Aware Linear Attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ff_expansion: int = 4,
        dropout_rate: float = 0.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_expansion = ff_expansion
        self.dropout_rate = dropout_rate
        self.eps = eps

    def build(self, input_shape):
        self.mha = MultiHeadNaLaAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            eps=self.eps,
            name="mha_nala",
        )
        self.ffn = FeedForward(
            d_model=self.d_model,
            expansion=self.ff_expansion,
            dropout_rate=self.dropout_rate,
            name="ffn",
        )
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_attn")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_ffn")
        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=None):
        # --- Multi-Head NaLa Attention + residual ---
        attn_out = self.mha(x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.ln1(x + attn_out)

        # --- Feed-Forward + residual ---
        ffn_out = self.ffn(x, training=training)
        ffn_out = self.drop2(ffn_out, training=training)
        x = self.ln2(x + ffn_out)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_expansion": self.ff_expansion,
            "dropout_rate": self.dropout_rate,
            "eps": self.eps,
        })
        return config


# ============================================================================
# 9. NaLaFormer Encoder  (chồng nhiều block)
# ============================================================================
class NaLaFormerEncoder(tf.keras.layers.Layer):
    """
    Stack N × NaLaFormerBlock.

    Input Embedding (patch / linear) → [NaLaFormerBlock] × depth → output
    """

    def __init__(
        self,
        d_model: int = 256,
        depth: int = 6,
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
            NaLaFormerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_expansion=self.ff_expansion,
                dropout_rate=self.dropout_rate,
                name=f"nala_block_{i}",
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


# ============================================================================
# 10. NaLaFormer Bottleneck  — Drop-in thay thế cho Transformer bottleneck
# ============================================================================
class NaLaFormerBottleneck(tf.keras.layers.Layer):
    """
    NaLaFormer Bottleneck cho kiến trúc U-Net / TransUNet.

    Thay thế Softmax Multi-Head Attention ở bottleneck bằng 
    Q Norm-Aware Linear Attention → giảm complexity từ O(N²) → O(N).

    Input:  feature map (B, H, W, C) từ encoder
    Output: feature map (B, H, W, C) đã qua NaLaFormer

    Luồng xử lý:
      1. Reshape (B, H, W, C) → (B, H*W, C)
      2. [Optional] Linear projection nếu C ≠ d_model
      3. NaLaFormer Encoder (depth blocks)
      4. [Optional] Project back
      5. Reshape → (B, H, W, C)
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
                self.d_model, use_bias=False, name="proj_in"
            )
            self.proj_out = tf.keras.layers.Dense(
                C, use_bias=False, name="proj_out"
            )

        self.encoder = NaLaFormerEncoder(
            d_model=self.d_model,
            depth=self.depth,
            num_heads=self.num_heads,
            ff_expansion=self.ff_expansion,
            dropout_rate=self.dropout_rate,
            name="nala_encoder",
        )
        super().build(input_shape)

    def call(self, x, training=None):
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        # Flatten spatial → sequence
        x = tf.reshape(x, [B, H * W, C])            # (B, N, C)

        if self.need_proj:
            x = self.proj_in(x)                       # (B, N, d_model)

        x = self.encoder(x, training=training)        # (B, N, d_model)

        if self.need_proj:
            x = self.proj_out(x)                      # (B, N, C)

        # Reshape lại spatial
        x = tf.reshape(x, [B, H, W, C])              # (B, H, W, C)
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


# ============================================================================
# TIỆN ÍCH:  Quick-build helper
# ============================================================================
def build_nalaformer_bottleneck(
    d_model: int = 256,
    depth: int = 4,
    num_heads: int = 8,
    ff_expansion: int = 4,
    dropout_rate: float = 0.1,
):
    """
    Hàm tiện ích tạo NaLaFormerBottleneck layer.

    Ví dụ sử dụng trong U-Net:
    >>> cnn_out = encoder(input_img)          # (B, 12, 12, 2048)
    >>> bottleneck = build_nalaformer_bottleneck(d_model=256, depth=4)
    >>> x = bottleneck(cnn_out)               # (B, 12, 12, 2048)
    """
    return NaLaFormerBottleneck(
        d_model=d_model,
        depth=depth,
        num_heads=num_heads,
        ff_expansion=ff_expansion,
        dropout_rate=dropout_rate,
        name="nalaformer_bottleneck",
    )


# ============================================================================
# SMOKE TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NaLaFormer Attention — Smoke Test")
    print("=" * 60)

    # --- Test 1: Single-head attention ---
    print("\n[1] QNormAwareLinearAttention (d=64)")
    attn = QNormAwareLinearAttention(d_model=64)
    dummy = tf.random.normal((2, 100, 64))
    out = attn(dummy)
    print(f"    Input:  {dummy.shape}  →  Output: {out.shape}")
    assert out.shape == (2, 100, 64)

    # --- Test 2: Multi-head attention ---
    print("\n[2] MultiHeadNaLaAttention (d=256, heads=8)")
    mha = MultiHeadNaLaAttention(d_model=256, num_heads=8)
    dummy2 = tf.random.normal((2, 196, 256))
    out2 = mha(dummy2)
    print(f"    Input:  {dummy2.shape}  →  Output: {out2.shape}")
    assert out2.shape == (2, 196, 256)

    # --- Test 3: Full block ---
    print("\n[3] NaLaFormerBlock (d=256, heads=8)")
    block = NaLaFormerBlock(d_model=256, num_heads=8)
    out3 = block(dummy2)
    print(f"    Input:  {dummy2.shape}  →  Output: {out3.shape}")
    assert out3.shape == (2, 196, 256)

    # --- Test 4: Bottleneck (U-Net style) ---
    print("\n[4] NaLaFormerBottleneck (spatial input 12×12×512)")
    bneck = build_nalaformer_bottleneck(d_model=128, depth=2, num_heads=4)
    spatial = tf.random.normal((1, 12, 12, 512))
    out4 = bneck(spatial)
    print(f"    Input:  {spatial.shape}  →  Output: {out4.shape}")
    assert out4.shape == (1, 12, 12, 512)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  ✅ All smoke tests passed!")
    print("=" * 60)

    # Đếm tham số
    bneck_full = build_nalaformer_bottleneck(d_model=256, depth=4, num_heads=8)
    _ = bneck_full(tf.random.normal((1, 12, 12, 2048)))
    total = sum(tf.keras.backend.count_params(w) for w in bneck_full.trainable_weights)
    print(f"\n  Bottleneck params (d=256, depth=4): {total:,}")

"""
Dual-Decoder Architecture (Region Decoder + Boundary Decoder + Fusion + Refinement)
with ResNet Baseline Backbone for Medical Image Segmentation.

Key Modules:
1. ResNet Baseline Encoder (ResNet34 / ResNet50V2)
2. Skip Connection Attention Gates (Channel + Spatial Attention tại mỗi skip)
3. Region Decoder Branch (Semantic Region Features)
4. Boundary Decoder Branch (Edge & Boundary Features)
5. Extensible Attention Hook Slots (for Log-Linear, Nalaformer, Multipole Attention)
6. Boundary-Guided Fusion Module
7. Residual Refinement Module
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from omegaconf import DictConfig


# =============================================================================
# SKIP CONNECTION ATTENTION GATE
# =============================================================================
class SkipAttentionGate(tf.keras.layers.Layer):
    """
    Attention Gate cho Skip Connection.

    Lọc đặc trưng từ encoder (skip) dựa trên tín hiệu từ decoder (gating signal),
    giúp decoder chỉ nhận thông tin liên quan từ encoder, loại bỏ nhiễu nền.

    Cơ chế:
        α = σ(W_ψ(ψ) + W_g(g) + b)     (Channel-wise attention)
        skip_out = skip × α

    Trong đó:
        - skip: feature map từ encoder (E_i)
        - g:    gating signal từ decoder (tầng bên dưới, đã upsample)

    Tham khảo: "Attention U-Net: Learning Where to Look for the Pancreas"
                (Oktay et al., 2018, arXiv:1804.03999)
    """

    def __init__(self, inter_channels=None, **kwargs):
        """
        Args:
            inter_channels: Số kênh trung gian cho bottleneck projection.
                            Nếu None, tự động = skip_channels // 2.
        """
        super().__init__(**kwargs)
        self.inter_channels = inter_channels

    def build(self, input_shape):
        # input_shape là list [skip_shape, gate_shape]
        skip_shape, gate_shape = input_shape
        skip_c = skip_shape[-1]
        gate_c = gate_shape[-1]
        inter_c = self.inter_channels or max(skip_c // 2, 1)

        # Projection cho skip signal
        self.W_skip = layers.Conv2D(
            inter_c, (1, 1), strides=(1, 1), padding='same',
            use_bias=True, name=f"{self.name}_W_skip"
        )

        # Projection cho gating signal
        self.W_gate = layers.Conv2D(
            inter_c, (1, 1), strides=(1, 1), padding='same',
            use_bias=True, name=f"{self.name}_W_gate"
        )

        # Attention coefficient projection
        self.psi = layers.Conv2D(
            1, (1, 1), strides=(1, 1), padding='same',
            use_bias=True, name=f"{self.name}_psi"
        )

        self.bn = layers.BatchNormalization(name=f"{self.name}_bn")

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: list of [skip_feature, gating_signal]
                skip_feature:   (B, H, W, C_skip)  — từ encoder
                gating_signal:  (B, H, W, C_gate)  — từ decoder (đã upsample về cùng H, W)
        Returns:
            attended_skip: (B, H, W, C_skip) — skip feature đã được lọc bởi attention
        """
        skip, gate = inputs

        # Additive attention
        x_skip = self.W_skip(skip)          # (B, H, W, inter_c)
        x_gate = self.W_gate(gate)          # (B, H, W, inter_c)

        # Cộng + ReLU
        combined = layers.Activation('relu')(x_skip + x_gate)

        # Attention map α ∈ [0, 1]
        alpha = self.psi(combined)           # (B, H, W, 1)
        alpha = layers.Activation('sigmoid')(alpha)

        # Lọc skip bằng attention
        attended = skip * alpha              # (B, H, W, C_skip)
        attended = self.bn(attended, training=training)
        return attended

    def get_config(self):
        config = super().get_config()
        config.update({"inter_channels": self.inter_channels})
        return config


# =============================================================================
# COMMON BLOCKS
# =============================================================================
def conv_block(x, filters, name_prefix="conv"):
    """Standard Conv-BN-ReLU Block"""
    x = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu2")(x)
    return x


def residual_refinement_block(x, filters, name_prefix="refine"):
    """Residual Refinement Block for fine-tuning boundaries and regions"""
    res = layers.Conv2D(filters, (1, 1), padding='same', name=f"{name_prefix}_res_proj")(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)

    x = layers.Add(name=f"{name_prefix}_add")([res, x])
    x = layers.Activation('relu', name=f"{name_prefix}_out")(x)
    return x


def apply_skip_attention(skip, gate, stage_idx, attn_type, prefix="reg"):
    """
    Ap dung co che Attention tai Skip Connection:
    - 'nalaformer' / 'nala': NaLaFormer Linear Attention (2026)
    - 'log_linear' / 'log': Log-Linear Attention (ICLR 2026)
    - 'multipole' / 'mutil': Multipole Attention (ICCV 2025)
    - 'oktay' / 'gate': Additive Gated Attention (Oktay et al., 2018)
    """
    attn_type = str(attn_type).lower()
    if attn_type in ["nalaformer", "nala"]:
        from .nalaformer_attention import build_nalaformer_bottleneck
        return build_nalaformer_bottleneck(
            d_model=128, depth=1, num_heads=4, name=f"{prefix}_skip_nala{stage_idx}"
        )(skip)
    elif attn_type in ["log_linear", "log", "loglinear"]:
        from .log_linear_attention import build_loglinear_bottleneck
        return build_loglinear_bottleneck(
            d_model=128, depth=1, num_heads=4, name=f"{prefix}_skip_log{stage_idx}"
        )(skip)
    elif attn_type in ["multipole", "mutil", "mano"]:
        from .multipole_attention import build_multipole_bottleneck
        return build_multipole_bottleneck(
            d_model=128, depth=1, num_heads=4, name=f"{prefix}_skip_multi{stage_idx}"
        )(skip)
    else:
        return SkipAttentionGate(name=f"{prefix}_skip_att{stage_idx}")([skip, gate])


# =============================================================================
# MAIN MODEL BUILDER
# =============================================================================
def build_dual_decoder_resnet(cfg: DictConfig):
    """
    Build Dual-Decoder ResNet Model with Skip Connection Attention Gates.

    Returns:
        tf.keras.Model with 3 outputs:
          1. 'region_output': Intermediate Region Mask
          2. 'boundary_output': Intermediate Boundary Mask
          3. 'refined_output': Final Refined Region Mask
    """
    img_size = cfg.INPUT.HEIGHT
    channels = cfg.INPUT.CHANNELS
    num_classes = cfg.OUTPUT.CLASSES
    backbone_type = getattr(cfg.MODEL.BACKBONE, "TYPE", "resnet50v2").lower()

    input_layer = layers.Input(shape=(img_size, img_size, channels), name="input_image")

    # =========================================================================
    # 1. ENCODER (ResNet Baseline)
    # =========================================================================
    if "34" in backbone_type:
        from .backbones import resnet34_backbone
        e1, e2, e3, e4, e5 = resnet34_backbone(input_layer)

    else:
        # Default ResNet50V2
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False, weights='imagenet', input_tensor=input_layer
        )
        e1 = base_model.get_layer("conv1_conv").output       # 192x192
        e2 = base_model.get_layer("conv2_block2_out").output # 96x96
        e3 = base_model.get_layer("conv3_block3_out").output # 48x48
        e4 = base_model.get_layer("conv4_block5_out").output # 24x24
        e5 = base_model.output                               # 12x12

    bottleneck = e5

    # =========================================================================
    # HOOK POINT 1: ATTENTION MODULE AT BOTTLENECK (OPTIONAL)
    # =========================================================================
    bottleneck_attn = str(getattr(cfg.MODEL, "BOTTLENECK_ATTENTION", "none")).lower()
    if bottleneck_attn in ["log_linear", "log", "loglinear"]:
        from .log_linear_attention import build_loglinear_bottleneck
        print("[INFO] Applying Log-Linear Bottleneck Attention (ICLR 2026)")
        bottleneck = build_loglinear_bottleneck(d_model=256, depth=2, num_heads=8, name="loglinear_bottleneck")(bottleneck)
    elif bottleneck_attn in ["multipole", "mutil", "mano"]:
        from .multipole_attention import build_multipole_bottleneck
        print("[INFO] Applying Multipole Bottleneck Attention (ICCV 2025 Workshop)")
        bottleneck = build_multipole_bottleneck(d_model=256, depth=2, num_heads=8, name="multipole_bottleneck")(bottleneck)
    elif bottleneck_attn in ["nalaformer", "nala"]:
        from .nalaformer_attention import build_nalaformer_bottleneck
        print("[INFO] Applying NaLaFormer Bottleneck Attention (2026)")
        bottleneck = build_nalaformer_bottleneck(d_model=256, depth=2, num_heads=8, name="nalaformer_bottleneck")(bottleneck)
    else:
        print("[INFO] Bottleneck Attention: None")

    # =========================================================================
    # HOOK POINT 2: ATTENTION MODULE AT SKIP CONNECTIONS (STAGES 3 & 4)
    # =========================================================================
    skip_stages = list(getattr(cfg.MODEL, "SKIP_ATTENTION_STAGES", [3, 4]))
    skip_attn_type = str(getattr(cfg.MODEL, "SKIP_ATTENTION_TYPE", "nalaformer")).lower()
    print(f"[INFO] Skip Attention Module: '{skip_attn_type}' active at stage(s): {skip_stages}")

    # =========================================================================
    # 2. REGION DECODER BRANCH
    # =========================================================================
    # --- Tầng 4: bottleneck -> 24x24, skip = e4 ---
    r4_up = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', name="reg_up4")(bottleneck)
    if 4 in skip_stages:
        e4_att = apply_skip_attention(e4, r4_up, 4, skip_attn_type, prefix="reg")
    else:
        e4_att = e4
    r4 = layers.Concatenate(name="reg_concat4")([r4_up, e4_att])
    r4 = conv_block(r4, 512, name_prefix="reg_block4")

    # --- Tầng 3: 24x24 -> 48x48, skip = e3 ---
    r3_up = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name="reg_up3")(r4)
    if 3 in skip_stages:
        e3_att = apply_skip_attention(e3, r3_up, 3, skip_attn_type, prefix="reg")
    else:
        e3_att = e3
    r3 = layers.Concatenate(name="reg_concat3")([r3_up, e3_att])
    r3 = conv_block(r3, 256, name_prefix="reg_block3")

    # --- Tầng 2: 48x48 -> 96x96, skip = e2 ---
    r2_up = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name="reg_up2")(r3)
    if 2 in skip_stages:
        e2_att = apply_skip_attention(e2, r2_up, 2, skip_attn_type, prefix="reg")
    else:
        e2_att = e2
    r2 = layers.Concatenate(name="reg_concat2")([r2_up, e2_att])
    r2 = conv_block(r2, 128, name_prefix="reg_block2")

    # --- Tầng 1: 96x96 -> 192x192, skip = e1 ---
    r1_up = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name="reg_up1")(r2)
    if 1 in skip_stages:
        e1_att = apply_skip_attention(e1, r1_up, 1, skip_attn_type, prefix="reg")
    else:
        e1_att = e1
    r1 = layers.Concatenate(name="reg_concat1")([r1_up, e1_att])
    f_region = conv_block(r1, 64, name_prefix="f_region_block")

    # Upsample lên đúng resolution ảnh gốc (192x192 -> 384x384)
    f_region_full = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name="reg_full_up")(f_region)
    f_region_full = conv_block(f_region_full, 32, name_prefix="f_region_full")

    activation_func = 'sigmoid' if num_classes == 1 else 'softmax'
    region_output = layers.Conv2D(num_classes, (1, 1), activation=activation_func, name="region_output")(f_region_full)

    # =========================================================================
    # 3. BOUNDARY DECODER BRANCH
    # =========================================================================
    # --- Tầng 4: bottleneck -> 24x24, skip = e4 ---
    b4_up = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name="bound_up4")(bottleneck)
    if 4 in skip_stages:
        e4_att_b = apply_skip_attention(e4, b4_up, 4, skip_attn_type, prefix="bound")
    else:
        e4_att_b = e4
    b4 = layers.Concatenate(name="bound_concat4")([b4_up, e4_att_b])
    b4 = conv_block(b4, 256, name_prefix="bound_block4")

    # --- Tầng 3: 24x24 -> 48x48, skip = e3 ---
    b3_up = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name="bound_up3")(b4)
    if 3 in skip_stages:
        e3_att_b = apply_skip_attention(e3, b3_up, 3, skip_attn_type, prefix="bound")
    else:
        e3_att_b = e3
    b3 = layers.Concatenate(name="bound_concat3")([b3_up, e3_att_b])
    b3 = conv_block(b3, 128, name_prefix="bound_block3")

    # --- Tầng 2: 48x48 -> 96x96, skip = e2 ---
    b2_up = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name="bound_up2")(b3)
    if 2 in skip_stages:
        e2_att_b = apply_skip_attention(e2, b2_up, 2, skip_attn_type, prefix="bound")
    else:
        e2_att_b = e2
    b2 = layers.Concatenate(name="bound_concat2")([b2_up, e2_att_b])
    b2 = conv_block(b2, 64, name_prefix="bound_block2")

    # --- Tầng 1: 96x96 -> 192x192, skip = e1 ---
    b1_up = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name="bound_up1")(b2)
    if 1 in skip_stages:
        e1_att_b = apply_skip_attention(e1, b1_up, 1, skip_attn_type, prefix="bound")
    else:
        e1_att_b = e1
    b1 = layers.Concatenate(name="bound_concat1")([b1_up, e1_att_b])
    f_boundary = conv_block(b1, 32, name_prefix="f_boundary_block")

    f_boundary_full = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name="bound_full_up")(f_boundary)
    f_boundary_full = conv_block(f_boundary_full, 32, name_prefix="f_boundary_full")

    boundary_output = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name="boundary_output")(f_boundary_full)

    # =========================================================================
    # 4. FUSION MODULE (Kết Hợp Đặc Trưng Region & Boundary)
    # =========================================================================
    # Cổng Attention Ranh Giới (Boundary Attention Gate)
    boundary_gate = layers.Conv2D(32, (1, 1), activation='sigmoid', name="boundary_gate")(f_boundary_full)
    f_region_gated = layers.Multiply(name="region_gated")([f_region_full, boundary_gate])

    fused_features = layers.Concatenate(name="fusion_concat")([f_region_full, f_region_gated, f_boundary_full])
    fused_features = layers.Conv2D(64, (3, 3), padding='same', name="fusion_conv1")(fused_features)
    fused_features = layers.BatchNormalization(name="fusion_bn1")(fused_features)
    fused_features = layers.Activation('relu', name="fusion_relu1")(fused_features)

    # =========================================================================
    # HOOK POINT: ATTENTION MODULE AT FUSION
    # =========================================================================
    # Vị trí chèn Attention thứ hai nếu muốn hướng chú ý vào vùng đặc trưng sau hợp nhất:
    # fused_features = AttentionLayer(...)(fused_features)
    # =========================================================================

    # =========================================================================
    # 5. REFINEMENT MODULE (Tinh Chỉnh Kết Quả Cuối Cùng)
    # =========================================================================
    refine_in = layers.Concatenate(name="refine_concat")([fused_features, region_output, boundary_output])
    refine_feat = residual_refinement_block(refine_in, 64, name_prefix="refine_block1")
    refine_feat = residual_refinement_block(refine_feat, 32, name_prefix="refine_block2")

    refined_output = layers.Conv2D(num_classes, (1, 1), activation=activation_func, name="refined_output")(refine_feat)

    # =========================================================================
    # 6. MODEL COMPOSITION (3 Outputs)
    # =========================================================================
    model = models.Model(
        inputs=input_layer,
        outputs=[region_output, boundary_output, refined_output],
        name="DualDecoder_ResNet"
    )

    return model

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
# Sử dụng ContextAttentionGate từ models.attention_gates thay thế các gate cũ.

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

def apply_skip_attention(skip, gate, stage_idx, branch_cfg, prefix="reg"):
    """
    Áp dụng Attention Gate tại Skip Connection.
    - skip: feature map từ encoder
    - gate: feature map từ decoder (sau upsample)
    - branch_cfg: dictionary config cho nhánh (chứa TYPE và STAGES)
    """
    stages = list(branch_cfg.get("STAGES", []))
    if stage_idx not in stages:
        return skip

    attn_type = str(branch_cfg.get("TYPE", "none")).lower()
    name = f"{prefix}_skipgate{stage_idx}"

    if attn_type in ["none", "", "null"]:
        return skip
    elif attn_type in ["oktay", "gate"]:
        context_type = "none"
    elif attn_type in ["nalaformer", "log_linear", "multipole"]:
        context_type = attn_type
    else:
        raise ValueError(f"SKIP_ATTENTION TYPE không hợp lệ: '{attn_type}'")

    from models.attention_gates import ContextAttentionGate
    out, _alpha = ContextAttentionGate(context_type=context_type, name=name)([skip, gate])
    return out


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
        bb_weights = getattr(cfg.MODEL.BACKBONE, "WEIGHTS", "imagenet")
        if isinstance(bb_weights, str):
            bb_weights = bb_weights.lower()
        e1, e2, e3, e4, e5 = resnet34_backbone(input_layer, weights=bb_weights)

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
    skip_cfg = getattr(cfg.MODEL, "SKIP_ATTENTION", None)
    
    if skip_cfg is None:
        old_type = str(getattr(cfg.MODEL, "SKIP_ATTENTION_TYPE", "oktay")).lower()
        old_stages = list(getattr(cfg.MODEL, "SKIP_ATTENTION_STAGES", [3, 4]))
        print(f"[WARNING] Cấu hình SKIP_ATTENTION cũ, dùng chung cho cả 2 nhánh. Type: {old_type}")
        region_skip_cfg = {"TYPE": old_type, "STAGES": old_stages}
        bound_skip_cfg = {"TYPE": old_type, "STAGES": old_stages}
    else:
        # Nếu dùng DictConfig, convert về dict hoặc dùng kiểu getattr
        def _get_dict(node):
            if hasattr(node, "keys"):
                return {k: node[k] for k in node.keys()}
            return node
        
        region_cfg_raw = getattr(skip_cfg, "REGION", {"TYPE": "none", "STAGES": []})
        bound_cfg_raw = getattr(skip_cfg, "BOUNDARY", {"TYPE": "none", "STAGES": []})
        
        region_skip_cfg = _get_dict(region_cfg_raw)
        bound_skip_cfg = _get_dict(bound_cfg_raw)

    print(f"[INFO] Skip REGION:   {region_skip_cfg.get('TYPE')}-gate @ {region_skip_cfg.get('STAGES')}")
    print(f"[INFO] Skip BOUNDARY: {bound_skip_cfg.get('TYPE')}-gate @ {bound_skip_cfg.get('STAGES')}")

    # =========================================================================
    # DECODER SETTINGS
    # =========================================================================
    is_dual_decoder = getattr(cfg.MODEL, "DUAL_DECODER", True)
    decoder_cfg = getattr(cfg.MODEL, "DECODER", {})
    decoder_type = str(decoder_cfg.get("TYPE", "unet")).lower()
    
    # =========================================================================
    # 2. REGION DECODER BRANCH
    # =========================================================================
    if decoder_type == "unet3plus":
        from models.unet3plus_decoder import build_unet3plus_decoder
        cat_ch_reg = decoder_cfg.get("CAT_CHANNELS_REGION", 64)
        f_region = build_unet3plus_decoder([e1, e2, e3, e4, e5], cat_ch_reg, prefix="reg", skip_cfg=region_skip_cfg)
    else:
        # --- Tầng 4: bottleneck -> 24x24, skip = e4 ---
        r4_up = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', name="reg_up4")(bottleneck)
        e4_att = apply_skip_attention(e4, r4_up, 4, region_skip_cfg, prefix="reg")
        r4 = layers.Concatenate(name="reg_concat4")([r4_up, e4_att])
        r4 = conv_block(r4, 512, name_prefix="reg_block4")

        # --- Tầng 3: 24x24 -> 48x48, skip = e3 ---
        r3_up = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name="reg_up3")(r4)
        e3_att = apply_skip_attention(e3, r3_up, 3, region_skip_cfg, prefix="reg")
        r3 = layers.Concatenate(name="reg_concat3")([r3_up, e3_att])
        r3 = conv_block(r3, 256, name_prefix="reg_block3")

        # --- Tầng 2: 48x48 -> 96x96, skip = e2 ---
        r2_up = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name="reg_up2")(r3)
        e2_att = apply_skip_attention(e2, r2_up, 2, region_skip_cfg, prefix="reg")
        r2 = layers.Concatenate(name="reg_concat2")([r2_up, e2_att])
        r2 = conv_block(r2, 128, name_prefix="reg_block2")

        # --- Tầng 1: 96x96 -> 192x192, skip = e1 ---
        r1_up = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name="reg_up1")(r2)
        e1_att = apply_skip_attention(e1, r1_up, 1, region_skip_cfg, prefix="reg")
        r1 = layers.Concatenate(name="reg_concat1")([r1_up, e1_att])
        f_region = conv_block(r1, 64, name_prefix="f_region_block")

    # Upsample lên đúng resolution ảnh gốc (192x192 -> 384x384)
    f_region_full = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name="reg_full_up")(f_region)
    f_region_full = conv_block(f_region_full, 32, name_prefix="f_region_full")

    activation_func = 'sigmoid' if num_classes == 1 else 'softmax'
    
    if not is_dual_decoder:
        # M0 Mode: only Region branch, name output 'refined_output' for compatibility
        refined_output = layers.Conv2D(num_classes, (1, 1), activation=activation_func, dtype='float32', name="refined_output")(f_region_full)
        model = models.Model(inputs=input_layer, outputs=refined_output, name="M0_Region_Only_ResNet")
        return model

    region_output = layers.Conv2D(num_classes, (1, 1), activation=activation_func, dtype='float32', name="region_output")(f_region_full)

    # =========================================================================
    # 3. BOUNDARY DECODER BRANCH
    # =========================================================================
    if decoder_type == "unet3plus":
        from models.unet3plus_decoder import build_unet3plus_decoder
        cat_ch_bnd = decoder_cfg.get("CAT_CHANNELS_BOUNDARY", 32)
        f_boundary = build_unet3plus_decoder([e1, e2, e3, e4, e5], cat_ch_bnd, prefix="bound", skip_cfg=bound_skip_cfg)
    else:
        # --- Tầng 4: bottleneck -> 24x24, skip = e4 ---
        b4_up = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name="bound_up4")(bottleneck)
        e4_att_b = apply_skip_attention(e4, b4_up, 4, bound_skip_cfg, prefix="bound")
        b4 = layers.Concatenate(name="bound_concat4")([b4_up, e4_att_b])
        b4 = conv_block(b4, 256, name_prefix="bound_block4")

        # --- Tầng 3: 24x24 -> 48x48, skip = e3 ---
        b3_up = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name="bound_up3")(b4)
        e3_att_b = apply_skip_attention(e3, b3_up, 3, bound_skip_cfg, prefix="bound")
        b3 = layers.Concatenate(name="bound_concat3")([b3_up, e3_att_b])
        b3 = conv_block(b3, 128, name_prefix="bound_block3")

        # --- Tầng 2: 48x48 -> 96x96, skip = e2 ---
        b2_up = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name="bound_up2")(b3)
        e2_att_b = apply_skip_attention(e2, b2_up, 2, bound_skip_cfg, prefix="bound")
        b2 = layers.Concatenate(name="bound_concat2")([b2_up, e2_att_b])
        b2 = conv_block(b2, 64, name_prefix="bound_block2")

        # --- Tầng 1: 96x96 -> 192x192, skip = e1 ---
        b1_up = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name="bound_up1")(b2)
        e1_att_b = apply_skip_attention(e1, b1_up, 1, bound_skip_cfg, prefix="bound")
        b1 = layers.Concatenate(name="bound_concat1")([b1_up, e1_att_b])
        f_boundary = conv_block(b1, 32, name_prefix="f_boundary_block")

    f_boundary_full = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name="bound_full_up")(f_boundary)
    f_boundary_full = conv_block(f_boundary_full, 32, name_prefix="f_boundary_full")

    boundary_output = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', dtype='float32', name="boundary_output")(f_boundary_full)

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

    refined_output = layers.Conv2D(num_classes, (1, 1), activation=activation_func, dtype='float32', name="refined_output")(refine_feat)

    # =========================================================================
    # 6. MODEL COMPOSITION (3 Outputs)
    # =========================================================================
    model = models.Model(
        inputs=input_layer,
        outputs=[region_output, boundary_output, refined_output],
        name="DualDecoder_ResNet"
    )

    return model

def get_gate_maps_model(model):
    """
    Tạo model trích xuất tất cả các attention maps (hệ số alpha) từ ContextAttentionGate.
    """
    from models.attention_gates import ContextAttentionGate
    gate_outputs = []
    for layer in model.layers:
        if isinstance(layer, ContextAttentionGate):
            gate_outputs.append(layer.output[1])
    
    if not gate_outputs:
        print("[WARNING] No ContextAttentionGate found in the model.")
        
    return models.Model(inputs=model.input, outputs=gate_outputs, name="AttentionGateMaps_Extractor")

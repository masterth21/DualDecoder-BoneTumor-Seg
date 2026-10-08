import tensorflow as tf
from tensorflow.keras import layers
from models.attention_gates import ContextAttentionGate

def custom_conv_block(x, filters, name_prefix):
    x = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu1")(x)
    return x

def build_unet3plus_decoder(enc, cat_channels, prefix, skip_cfg):
    e1, e2, e3, e4, e5 = enc
    d5 = e5
    
    stages = list(skip_cfg.get("STAGES", []))
    attn_type = str(skip_cfg.get("TYPE", "none")).lower()
    
    def get_gate(stage_idx, e_i, d_i_plus_1):
        if stage_idx not in stages or attn_type in ["none", "", "null"]:
            return e_i
        g_i = layers.UpSampling2D(2, interpolation='bilinear', name=f"{prefix}_gate_up{stage_idx}")(d_i_plus_1)
        name = f"{prefix}_skipgate{stage_idx}"
        
        ctx_type = "none"
        if attn_type in ["nalaformer", "log_linear", "multipole"]:
            ctx_type = attn_type
            
        out, _ = ContextAttentionGate(context_type=ctx_type, name=name)([e_i, g_i])
        return out

    # d4
    e1_d4 = layers.MaxPool2D(8, name=f"{prefix}_e1_d4_pool")(e1)
    e1_d4 = custom_conv_block(e1_d4, cat_channels, f"{prefix}_e1_d4")
    e2_d4 = layers.MaxPool2D(4, name=f"{prefix}_e2_d4_pool")(e2)
    e2_d4 = custom_conv_block(e2_d4, cat_channels, f"{prefix}_e2_d4")
    e3_d4 = layers.MaxPool2D(2, name=f"{prefix}_e3_d4_pool")(e3)
    e3_d4 = custom_conv_block(e3_d4, cat_channels, f"{prefix}_e3_d4")
    
    e4_att = get_gate(4, e4, d5)
    e4_d4 = custom_conv_block(e4_att, cat_channels, f"{prefix}_e4_d4")
    
    d5_d4 = layers.UpSampling2D(2, interpolation='bilinear', name=f"{prefix}_d5_d4_up")(d5)
    d5_d4 = custom_conv_block(d5_d4, cat_channels, f"{prefix}_d5_d4")
    
    d4 = layers.Concatenate(name=f"{prefix}_d4_concat")([e1_d4, e2_d4, e3_d4, e4_d4, d5_d4])
    d4 = custom_conv_block(d4, 5 * cat_channels, f"{prefix}_d4")

    # d3
    e1_d3 = layers.MaxPool2D(4, name=f"{prefix}_e1_d3_pool")(e1)
    e1_d3 = custom_conv_block(e1_d3, cat_channels, f"{prefix}_e1_d3")
    e2_d3 = layers.MaxPool2D(2, name=f"{prefix}_e2_d3_pool")(e2)
    e2_d3 = custom_conv_block(e2_d3, cat_channels, f"{prefix}_e2_d3")
    
    e3_att = get_gate(3, e3, d4)
    e3_d3 = custom_conv_block(e3_att, cat_channels, f"{prefix}_e3_d3")
    
    d4_d3 = layers.UpSampling2D(2, interpolation='bilinear', name=f"{prefix}_d4_d3_up")(d4)
    d4_d3 = custom_conv_block(d4_d3, cat_channels, f"{prefix}_d4_d3")
    d5_d3 = layers.UpSampling2D(4, interpolation='bilinear', name=f"{prefix}_d5_d3_up")(d5)
    d5_d3 = custom_conv_block(d5_d3, cat_channels, f"{prefix}_d5_d3")
    
    d3 = layers.Concatenate(name=f"{prefix}_d3_concat")([e1_d3, e2_d3, e3_d3, d4_d3, d5_d3])
    d3 = custom_conv_block(d3, 5 * cat_channels, f"{prefix}_d3")

    # d2
    e1_d2 = layers.MaxPool2D(2, name=f"{prefix}_e1_d2_pool")(e1)
    e1_d2 = custom_conv_block(e1_d2, cat_channels, f"{prefix}_e1_d2")
    
    e2_att = get_gate(2, e2, d3)
    e2_d2 = custom_conv_block(e2_att, cat_channels, f"{prefix}_e2_d2")
    
    d3_d2 = layers.UpSampling2D(2, interpolation='bilinear', name=f"{prefix}_d3_d2_up")(d3)
    d3_d2 = custom_conv_block(d3_d2, cat_channels, f"{prefix}_d3_d2")
    d4_d2 = layers.UpSampling2D(4, interpolation='bilinear', name=f"{prefix}_d4_d2_up")(d4)
    d4_d2 = custom_conv_block(d4_d2, cat_channels, f"{prefix}_d4_d2")
    d5_d2 = layers.UpSampling2D(8, interpolation='bilinear', name=f"{prefix}_d5_d2_up")(d5)
    d5_d2 = custom_conv_block(d5_d2, cat_channels, f"{prefix}_d5_d2")
    
    d2 = layers.Concatenate(name=f"{prefix}_d2_concat")([e1_d2, e2_d2, d3_d2, d4_d2, d5_d2])
    d2 = custom_conv_block(d2, 5 * cat_channels, f"{prefix}_d2")

    # d1
    e1_att = get_gate(1, e1, d2)
    e1_d1 = custom_conv_block(e1_att, cat_channels, f"{prefix}_e1_d1")
    
    d2_d1 = layers.UpSampling2D(2, interpolation='bilinear', name=f"{prefix}_d2_d1_up")(d2)
    d2_d1 = custom_conv_block(d2_d1, cat_channels, f"{prefix}_d2_d1")
    d3_d1 = layers.UpSampling2D(4, interpolation='bilinear', name=f"{prefix}_d3_d1_up")(d3)
    d3_d1 = custom_conv_block(d3_d1, cat_channels, f"{prefix}_d3_d1")
    d4_d1 = layers.UpSampling2D(8, interpolation='bilinear', name=f"{prefix}_d4_d1_up")(d4)
    d4_d1 = custom_conv_block(d4_d1, cat_channels, f"{prefix}_d4_d1")
    d5_d1 = layers.UpSampling2D(16, interpolation='bilinear', name=f"{prefix}_d5_d1_up")(d5)
    d5_d1 = custom_conv_block(d5_d1, cat_channels, f"{prefix}_d5_d1")
    
    d1 = layers.Concatenate(name=f"{prefix}_d1_concat")([e1_d1, d2_d1, d3_d1, d4_d1, d5_d1])
    d1 = custom_conv_block(d1, 5 * cat_channels, f"{prefix}_d1")

    # d1 to full resolution (192^2 -> 384^2)
    f_full = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name=f"{prefix}_full_up")(d1)
    f_full = custom_conv_block(f_full, 32, f"{prefix}_full_c1")
    f_full = custom_conv_block(f_full, 32, f"{prefix}_full_c2")
    
    return f_full

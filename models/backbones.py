"""
Unet3+ backbones
"""
import tensorflow as tf
import tensorflow.keras as k
from .unet3plus_utils import conv_block


def vgg16_backbone(input_layer, ):
    """ VGG-16 backbone as encoder for UNet3P """

    base_model = tf.keras.applications.VGG16(
        input_tensor=input_layer,
        weights=None,
        include_top=False
    )

    # block 1
    e1 = base_model.get_layer("block1_conv2").output  # 320, 320, 64
    # block 2
    e2 = base_model.get_layer("block2_conv2").output  # 160, 160, 128
    # block 3
    e3 = base_model.get_layer("block3_conv3").output  # 80, 80, 256
    # block 4
    e4 = base_model.get_layer("block4_conv3").output  # 40, 40, 512
    # block 5
    e5 = base_model.get_layer("block5_conv3").output  # 20, 20, 512

    return [e1, e2, e3, e4, e5]


def vgg19_backbone(input_layer, ):
    """ VGG-19 backbone as encoder for UNet3P """

    base_model = tf.keras.applications.VGG19(
        input_tensor=input_layer,
        weights=None,
        include_top=False
    )

    # block 1
    e1 = base_model.get_layer("block1_conv2").output  # 320, 320, 64
    # block 2
    e2 = base_model.get_layer("block2_conv2").output  # 160, 160, 128
    # block 3
    e3 = base_model.get_layer("block3_conv4").output  # 80, 80, 256
    # block 4
    e4 = base_model.get_layer("block4_conv4").output  # 40, 40, 512
    # block 5
    e5 = base_model.get_layer("block5_conv4").output  # 20, 20, 512

    return [e1, e2, e3, e4, e5]


def unet3plus_backbone(input_layer, filters):
    """ UNet3+ own backbone """
    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])  # 320*320*64
    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
    e2 = conv_block(e2, filters[1])  # 160*160*128
    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
    e3 = conv_block(e3, filters[2])  # 80*80*256
    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
    e4 = conv_block(e4, filters[3])  # 40*40*512
    # block 5, bottleneck layer
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
    e5 = conv_block(e5, filters[4])  # 20*20*1024

    return [e1, e2, e3, e4, e5]
# def resnet34_backbone(input_layer):
#     """ ResNet34 backbone as encoder cho UNet3P (Sử dụng thư viện lõi classification_models) """
#     from classification_models.tfkeras import Classifiers
#     import tensorflow as tf

#     # 1. Lấy kiến trúc ResNet34 chuẩn từ thư viện lõi
#     ResNet34, _ = Classifiers.get('resnet34')

#     # 2. Khởi tạo base_model và truyền trực tiếp input_layer vào (Đồ thị được nối liền)
#     base_model = ResNet34(input_tensor=input_layer, weights='imagenet', include_top=False)

#     # 3. Trích xuất 5 trạm đặc trưng với tên layer chuẩn xác
#     e1 = base_model.get_layer("relu0").output           
#     e2 = base_model.get_layer("stage2_unit1_relu1").output 
#     e3 = base_model.get_layer("stage3_unit1_relu1").output 
#     e4 = base_model.get_layer("stage4_unit1_relu1").output 
#     e5 = base_model.get_layer("relu1").output           

#     return [e1, e2, e3, e4, e5]

def resnet34_backbone(input_layer):
    """ ResNet34 backbone as encoder cho UNet3P (Sử dụng thư viện lõi classification_models) """
    from classification_models.tfkeras import Classifiers
    import tensorflow as tf

    # 1. Lấy kiến trúc ResNet34 chuẩn từ thư viện lõi
    ResNet34, _ = Classifiers.get('resnet34')

    # 2. Khởi tạo base_model, TẮT tải tự động ImageNet
    base_model = ResNet34(input_tensor=input_layer, weights=None, include_top=False)

    # ==============================================================
    # BƯỚC CẤY GHÉP: Nạp trọng số chuyên gia MURA (Transfer Learning)
    # ==============================================================
    import os
    mura_weights_path = 'mura_resnet34_best_weights.h5' if os.path.exists('mura_resnet34_best_weights.h5') else '/workspace/unet3p/mura_resnet34_best_weights.h5'
    if os.path.exists(mura_weights_path):
        try:
            base_model.load_weights(mura_weights_path, by_name=True, skip_mismatch=True)
            print("[INFO] Da tai thanh cong trong so chuyen gia MURA vao BACKBONE RESNET34!")
        except Exception as e:
            print(f"[WARNING] Khong the tai trong so MURA: {e}")
    else:
        print(f"[INFO] Khong tim thay file {mura_weights_path}. ResNet34 khoi tao mac dinh.")
    # ==============================================================

    # 3. Trích xuất 5 trạm đặc trưng với tên layer chuẩn xác
    e1 = base_model.get_layer("relu0").output            
    e2 = base_model.get_layer("stage2_unit1_relu1").output 
    e3 = base_model.get_layer("stage3_unit1_relu1").output 
    e4 = base_model.get_layer("stage4_unit1_relu1").output 
    e5 = base_model.get_layer("relu1").output            

    return [e1, e2, e3, e4, e5]
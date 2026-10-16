import sys
sys.path.append('.')
from utils.general_utils import get_data_paths
from data_generators import data_generator
from omegaconf import OmegaConf
import numpy as np

cfg = OmegaConf.load('configs/config.yaml')
cfg.WORK_DIR = '.'
cfg.DATASET.TRAIN.IMAGES_PATH = 'D:/TumorBone/data/BTXRD_NEW/Split/train/images'
cfg.DATASET.TRAIN.MASK_PATH = 'D:/TumorBone/data/BTXRD_NEW/Split/train/masks'

print("Checking Split/train")
paths = get_data_paths(cfg, 'TRAIN', True)
print(f"Split/train: {len(paths[0])} images, {len(paths[1])} masks")

print("Checking Split_full/test")
cfg.DATASET.TEST.IMAGES_PATH = 'D:/TumorBone/data/BTXRD_NEW/Split_full/test/images'
cfg.DATASET.TEST.MASK_PATH = 'D:/TumorBone/data/BTXRD_NEW/Split_full/test/masks'
paths = get_data_paths(cfg, 'TEST', True)
print(f"Split_full/test: {len(paths[0])} images, {len(paths[1])} masks")

gen = data_generator.get_data_generator(cfg, "TRAIN", None)
for x, y in gen:
    if isinstance(y, dict):
        y = y['refined_output']
    print("Batch X shape:", x.shape)
    print("Batch Y shape:", y.shape)
    print("Unique mask values:", np.unique(y))
    break

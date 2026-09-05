"""
Prediction script used to visualize model output
"""
import os
import hydra
from omegaconf import DictConfig

from data_generators import tf_data_generator
from utils.general_utils import join_paths, suppress_warnings
from utils.images_utils import display
from utils.images_utils import postprocess_mask, denormalize_mask
from models.model import prepare_model


def predict(cfg: DictConfig):
    """
    Predict and visualize given data
    """

    # suppress TensorFlow and DALI warnings
    suppress_warnings()

    # set batch size to one
    cfg.HYPER_PARAMETERS.BATCH_SIZE = 1

    # data generator
    val_generator = tf_data_generator.DataGenerator(cfg, mode="VAL")

    # create model
    model = prepare_model(cfg)

    # weights model path (auto-detect newest .weights.h5, .keras, or .hdf5)
    checkpoint_path = getattr(cfg, "CHECKPOINT_PATH", None)
    ckpt_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        import glob
        pattern = os.path.join(ckpt_dir, "*model*")
        found = [f for f in glob.glob(pattern) if f.endswith(('.weights.h5', '.keras', '.hdf5', '.h5'))]
        if found:
            found.sort(key=os.path.getmtime, reverse=True)
            checkpoint_path = found[0]
        else:
            checkpoint_path = join_paths(ckpt_dir, f"{cfg.MODEL.WEIGHTS_FILE_NAME}.weights.h5")

    print(f"[INFO] Loading model weights from: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), \
        f"Model weight file does not exist at:\n{checkpoint_path}\nPlease train the model first!"

    # load model weights
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)

    output_dir = join_paths(cfg.WORK_DIR, "outputs", "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # check mask are available or not
    mask_available = True
    if cfg.DATASET.VAL.MASK_PATH is None or \
            str(cfg.DATASET.VAL.MASK_PATH).lower() == "none":
        mask_available = False

    max_samples = int(getattr(cfg, "NUM_SAMPLES", 10))
    showed_images = 0
    print(f"[INFO] Predicting and saving {max_samples} sample visualizations to: {output_dir}")

    for batch_data in val_generator:  # for each batch
        batch_images = batch_data[0]
        if mask_available:
            batch_mask = batch_data[1]

        # make prediction on batch
        batch_predictions = model.predict_on_batch(batch_images)
        if len(model.outputs) > 1:
            if cfg.MODEL.TYPE == "dual_decoder_resnet":
                # Index 2 corresponds to the final refined region output for dual_decoder_resnet
                batch_predictions = batch_predictions[2]
            else:
                batch_predictions = batch_predictions[0]

        for index in range(len(batch_images)):
            image = batch_images[index]  # for each image
            if cfg.SHOW_CENTER_CHANNEL_IMAGE:
                # for UNet3+ show only center channel as image
                image = image[:, :, 1]

            # do postprocessing on predicted mask
            prediction = batch_predictions[index]
            prediction = postprocess_mask(prediction, cfg.OUTPUT.CLASSES)
            # denormalize mask for better visualization
            prediction = denormalize_mask(prediction, cfg.OUTPUT.CLASSES)

            save_file = os.path.join(output_dir, f"sample_{showed_images + 1:03d}.png")

            if mask_available:
                mask = batch_mask[index]
                mask = postprocess_mask(mask, cfg.OUTPUT.CLASSES)
                mask = denormalize_mask(mask, cfg.OUTPUT.CLASSES)
                display([image, mask, prediction], show_true_mask=True, save_path=save_file)
            else:
                display([image, prediction], show_true_mask=False, save_path=save_file)

            showed_images += 1
            print(f"  ✓ Saved: {save_file}")
            if showed_images >= max_samples:
                break

        if showed_images >= max_samples:
            break

    print(f"\n[SUCCESS] Completed saving {showed_images} prediction images in: {output_dir}\n")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to prediction method
    """
    predict(cfg)


if __name__ == "__main__":
    main()

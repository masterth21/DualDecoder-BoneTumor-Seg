import sys
import time
import tensorflow as tf


class CompactProgressCallback(tf.keras.callbacks.Callback):
    """
    Thanh tiến trình gọn gàng trên 1 dòng duy nhất trong lúc huấn luyện (in-place update).
    Khi kết thúc epoch, in ĐÚNG 1 DÒNG tóm tắt kết quả (Train/Val Loss, Dice Tổng, Lành, Ác, IoU, Prec, Rec, Thời gian).
    Tránh việc terminal bị tràn dòng sinh ra 300-400 dòng log mỗi epoch.
    """

    def __init__(self, total_steps, epochs, metric_prefix="refined_output_"):
        super().__init__()
        self.total_steps = max(1, total_steps)
        self.epochs = epochs
        self.epoch_start_time = None
        self.current_epoch = 1
        self.metric_prefix = metric_prefix

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.current_epoch = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        step = batch + 1
        elapsed = time.time() - self.epoch_start_time
        speed = elapsed / step if step > 0 else 0
        eta = max(0, speed * (self.total_steps - step))
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta)) if eta >= 3600 else time.strftime("%M:%S", time.gmtime(eta))

        pct = int(step / self.total_steps * 100)
        bar_len = 20
        filled = int(bar_len * step / self.total_steps)
        bar = "=" * filled + (">" if filled < bar_len else "") + "." * max(0, (bar_len - filled - 1 if filled < bar_len else 0))

        loss = logs.get('loss', 0.0)
        reg_loss = logs.get('region_output_loss', 0.0)
        bnd_loss = logs.get('boundary_output_loss', 0.0)
        ref_loss = logs.get('refined_output_loss', 0.0)
        
        refined_dice = logs.get(f'{self.metric_prefix}dice_coef', 0.0)
        dice_b = logs.get(f'{self.metric_prefix}dice_benign', 0.0)
        dice_m = logs.get(f'{self.metric_prefix}dice_malignant', 0.0)
        iou = logs.get(f'{self.metric_prefix}iou', 0.0)
        prec = logs.get(f'{self.metric_prefix}precision', 0.0)
        rec = logs.get(f'{self.metric_prefix}recall', 0.0)

        # Thanh tiến trình gọn, không bao giờ bị wrap trên terminal
        msg = (f"\rEp {self.current_epoch:03d}/{self.epochs} [{bar}] {step}/{self.total_steps} | "
               f"L:{loss:.3f}(R:{reg_loss:.2f}/B:{bnd_loss:.2f}/F:{ref_loss:.2f}) | "
               f"D:{refined_dice:.3f}(B:{dice_b:.3f}/M:{dice_m:.3f}) | "
               f"IoU:{iou:.3f} | P:{prec:.3f} | R:{rec:.3f}")
        sys.stdout.write(msg)
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.epoch_start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed)) if elapsed >= 3600 else time.strftime("%M:%S", time.gmtime(elapsed))

        train_loss = logs.get('loss', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        t_reg = logs.get('region_output_loss', 0.0)
        t_bnd = logs.get('boundary_output_loss', 0.0)
        t_ref = logs.get('refined_output_loss', 0.0)
        v_reg = logs.get('val_region_output_loss', 0.0)
        v_bnd = logs.get('val_boundary_output_loss', 0.0)
        v_ref = logs.get('val_refined_output_loss', 0.0)

        train_dice = logs.get(f'{self.metric_prefix}dice_coef', 0.0)
        train_dice_b = logs.get(f'{self.metric_prefix}dice_benign', 0.0)
        train_dice_m = logs.get(f'{self.metric_prefix}dice_malignant', 0.0)
        train_iou = logs.get(f'{self.metric_prefix}iou', 0.0)
        train_prec = logs.get(f'{self.metric_prefix}precision', 0.0)
        train_rec = logs.get(f'{self.metric_prefix}recall', 0.0)

        val_dice = logs.get(f'val_{self.metric_prefix}dice_coef', 0.0)
        val_dice_b = logs.get(f'val_{self.metric_prefix}dice_benign', 0.0)
        val_dice_m = logs.get(f'val_{self.metric_prefix}dice_malignant', 0.0)
        val_iou = logs.get(f'val_{self.metric_prefix}iou', 0.0)
        val_prec = logs.get(f'val_{self.metric_prefix}precision', 0.0)
        val_rec = logs.get(f'val_{self.metric_prefix}recall', 0.0)

        sys.stdout.write("\r" + " " * 160 + "\r")
        summary = (
            f"[Epoch {epoch + 1:03d}/{self.epochs:03d}] "
            f"Loss(Train/Val): {train_loss:.3f}/{val_loss:.3f} "
            f"| Dice(T/V): {train_dice:.3f}/{val_dice:.3f} "
            f"| DiceB(T/V): {train_dice_b:.3f}/{val_dice_b:.3f} "
            f"| DiceM(T/V): {train_dice_m:.3f}/{val_dice_m:.3f} "
            f"| IoU(T/V): {train_iou:.3f}/{val_iou:.3f} "
            f"| Prec(T/V): {train_prec:.3f}/{val_prec:.3f} "
            f"| Rec(T/V): {train_rec:.3f}/{val_rec:.3f} "
            f"| Time: {time_str}"
        )
        print(summary)
        sys.stdout.flush()

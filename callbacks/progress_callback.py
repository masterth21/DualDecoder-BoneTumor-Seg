import sys
import time
import tensorflow as tf


class CompactProgressCallback(tf.keras.callbacks.Callback):
    """
    Thanh tiáº¿n trÃ¬nh gá»n gÃ ng trÃªn 1 dÃ²ng duy nháº¥t trong lÃºc huáº¥n luyá»‡n (in-place update).
    Khi káº¿t thÃºc epoch, in ÄÃšNG 1 DÃ’NG tÃ³m táº¯t káº¿t quáº£ (Train/Val Loss, Train/Val Dice, Thá»i gian).
    TrÃ¡nh viá»‡c terminal bá»‹ trÃ n dÃ²ng sinh ra 300-400 dÃ²ng log má»—i epoch.
    """

    def __init__(self, total_steps, epochs):
        super().__init__()
        self.total_steps = max(1, total_steps)
        self.epochs = epochs
        self.epoch_start_time = None
        self.current_epoch = 1

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
        refined_dice = logs.get('refined_output_dice_coef', 0.0)

        # Thanh tiáº¿n trÃ¬nh gá»n ~75 kÃ½ tá»±, khÃ´ng bao giá» bá»‹ wrap trÃªn terminal
        msg = f"\rEpoch {self.current_epoch:03d}/{self.epochs} [{bar}] {step}/{self.total_steps} ({pct:2d}%) | ETA: {eta_str} | Loss: {loss:.4f} | Dice: {refined_dice:.4f}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.epoch_start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed)) if elapsed >= 3600 else time.strftime("%M:%S", time.gmtime(elapsed))

        train_loss = logs.get('loss', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        train_dice = logs.get('refined_output_dice_coef', 0.0)
        val_dice = logs.get('val_refined_output_dice_coef', 0.0)

        # XÃ³a dÃ²ng progress vÃ  in ÄÃšNG 1 DÃ’NG káº¿t quáº£ chá»‘t cá»§a Epoch
        sys.stdout.write("\r" + " " * 95 + "\r")
        summary = (
            f"[Epoch {epoch + 1:03d}/{self.epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Time: {time_str}"
        )
        print(summary)
        sys.stdout.flush()

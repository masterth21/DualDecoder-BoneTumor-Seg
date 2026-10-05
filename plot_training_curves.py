import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

# Tùy chỉnh thẩm mỹ theo chuẩn ấn phẩm khoa học / báo cáo
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


def find_column(df, candidates):
    """Tìm cột phù hợp trong dataframe dựa trên danh sách tên gợi ý."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def plot_history(
    log_file=None,
    model_type="dual_decoder_resnet",
    checkpoint_dir="checkpoint",
    output_dir="outputs/training_plots",
    show_plots=False
):
    """
    Hàm vẽ đồ thị huấn luyện từ file log CSV:
    1. Đồ thị Loss (Train Loss vs Val Loss)
    2. Đồ thị Dice Score (Overall, Benign, Malignant)
    3. Đồ thị IoU, Precision, Recall
    4. Bảng Dashboard tổng hợp 4 ô (2x2) đẹp mắt cho báo cáo
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Tự động tìm file log phù hợp nếu không truyền trực tiếp
    if not log_file or not os.path.exists(log_file):
        search_patterns = [
            os.path.join(checkpoint_dir, f"training_logs_{model_type}_*.csv"),
            os.path.join(checkpoint_dir, f"training_logs_*{model_type}*.csv"),
            os.path.join(checkpoint_dir, f"training_detailed_logs_{model_type}.csv"),
            os.path.join(checkpoint_dir, f"*{model_type}*.csv"),
            os.path.join(checkpoint_dir, "training_logs_*.csv"),
            os.path.join(checkpoint_dir, "*.csv"),
        ]

        found_files = []
        for pat in search_patterns:
            matches = glob.glob(pat)
            if matches:
                # Lọc bỏ file chi tiết test/eval không phải train log
                valid = [f for f in matches if not f.endswith("all_evaluations_history.csv") and not f.endswith("prediction_details_test.csv")]
                if valid:
                    found_files.extend(valid)
                    break

        if found_files:
            # Lấy file mới nhất
            found_files.sort(key=os.path.getmtime, reverse=True)
            log_file = found_files[0]
        else:
            raise FileNotFoundError(
                f"Không tìm thấy file log CSV nào cho model '{model_type}' trong thư mục '{checkpoint_dir}'!\n"
                f"Vui lòng chỉ định đường dẫn chính xác bằng tham số LOG_FILE='duong_dan_file.csv'."
            )

    print("\n" + "=" * 80)
    print("📈 VẼ ĐỒ THỊ QUÁ TRÌNH HUẤN LUYỆN (TRAINING CURVES)")
    print("=" * 80)
    print(f"Mô hình       : {model_type}")
    print(f"File log CSV  : {log_file}")
    print(f"Thư mục lưu ảnh: {output_dir}")
    print("=" * 80 + "\n")

    df = pd.read_csv(log_file)

    # Loại bỏ các dòng text bị lặp header (nếu có)
    if 'epoch' in df.columns:
        df = df[pd.to_numeric(df['epoch'], errors='coerce').notnull()]
        epochs = df['epoch'].astype(int) + 1
    elif 'Epoch' in df.columns:
        df = df[pd.to_numeric(df['Epoch'], errors='coerce').notnull()]
        epochs = df['Epoch'].astype(int)
    else:
        epochs = range(1, len(df) + 1)

    # 2. Nhận diện các cột
    col_t_loss = find_column(df, ['loss', 'train_loss', 'refined_output_loss'])
    col_v_loss = find_column(df, ['val_loss', 'val_refined_output_loss'])

    col_t_dice = find_column(df, ['refined_output_dice_coef', 'dice_coef', 'train_dice', 'dice'])
    col_v_dice = find_column(df, ['val_refined_output_dice_coef', 'val_dice_coef', 'val_dice', 'Dice_Overall'])
    col_v_dice_b = find_column(df, ['val_refined_output_dice_benign', 'val_dice_benign', 'Dice_Benign(Lanh)'])
    col_v_dice_m = find_column(df, ['val_refined_output_dice_malignant', 'val_dice_malignant', 'Dice_Malignant(Ac)'])

    col_v_iou = find_column(df, ['val_refined_output_iou', 'val_iou', 'iou', 'IoU_Overall'])
    col_v_prec = find_column(df, ['val_refined_output_precision', 'val_precision', 'Precision_Overall'])
    col_v_rec = find_column(df, ['val_refined_output_recall', 'val_recall', 'Recall_Overall'])
    col_lr = find_column(df, ['lr', 'learning_rate'])

    # =========================================================================
    # 3. VẼ TỪNG ĐỒ THỊ RIÊNG LẺ
    # =========================================================================

    # 3.1 Đồ thị LOSS
    if col_t_loss and col_v_loss:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, df[col_t_loss], label='Training Loss', color='#1f77b4', linewidth=2.0)
        ax.plot(epochs, df[col_v_loss], label='Validation Loss', color='#d62728', linewidth=2.0, linestyle='--')
        ax.set_title(f'Loss Convergence - {model_type}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss Value', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
        loss_path = os.path.join(output_dir, f"{model_type}_loss_curve.png")
        plt.savefig(loss_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Đã lưu biểu đồ Loss: {loss_path}")

    # 3.2 Đồ thị DICE SCORE
    if col_v_dice:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        if col_t_dice:
            ax.plot(epochs, df[col_t_dice], label='Train Dice', color='#7f7f7f', linewidth=1.5, alpha=0.7)
        ax.plot(epochs, df[col_v_dice], label='Val Dice (Overall)', color='#2ca02c', linewidth=2.5)

        if col_v_dice_b and col_v_dice_b in df.columns:
            ax.plot(epochs, df[col_v_dice_b], label='Val Dice (U Lành / Benign)', color='#1f77b4', linewidth=1.8, linestyle=':')
        if col_v_dice_m and col_v_dice_m in df.columns:
            ax.plot(epochs, df[col_v_dice_m], label='Val Dice (U Ác / Malignant)', color='#ff7f0e', linewidth=1.8, linestyle='-.')

        best_epoch_idx = df[col_v_dice].idxmax()
        best_epoch = epochs.iloc[best_epoch_idx]
        best_dice = df[col_v_dice].iloc[best_epoch_idx]
        ax.scatter([best_epoch], [best_dice], color='red', s=70, zorder=5)
        ax.annotate(
            f'Best: {best_dice:.4f} (Ep {best_epoch})',
            xy=(best_epoch, best_dice),
            xytext=(best_epoch - len(epochs)*0.15, best_dice - 0.08),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
            fontweight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6)
        )

        ax.set_title(f'Dice Score Progression - {model_type}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Dice Coefficient', fontsize=11)
        ax.set_ylim(bottom=0.0, top=min(1.0, max(best_dice + 0.15, 0.85)))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
        dice_path = os.path.join(output_dir, f"{model_type}_dice_curve.png")
        plt.savefig(dice_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Đã lưu biểu đồ Dice: {dice_path}")

    # =========================================================================
    # 4. VẼ BẢNG DASHBOARD TỔNG HỢP 2x2 (Chuẩn đưa vào bài báo / slide)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.suptitle(f'Training & Validation Performance Dashboard - {model_type}', fontsize=16, fontweight='bold', y=0.98)

    # Ô (0, 0): Loss
    if col_t_loss and col_v_loss:
        axes[0, 0].plot(epochs, df[col_t_loss], label='Training Loss', color='#1f77b4', linewidth=2.0)
        axes[0, 0].plot(epochs, df[col_v_loss], label='Validation Loss', color='#d62728', linewidth=2.0, linestyle='--')
        axes[0, 0].set_title('A. Training vs Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, linestyle=':', alpha=0.6)
        axes[0, 0].legend()

    # Ô (0, 1): Dice Score
    if col_v_dice:
        if col_t_dice:
            axes[0, 1].plot(epochs, df[col_t_dice], label='Train Dice', color='#7f7f7f', linewidth=1.5, alpha=0.7)
        axes[0, 1].plot(epochs, df[col_v_dice], label='Val Dice (Overall)', color='#2ca02c', linewidth=2.5)
        if col_v_dice_b and col_v_dice_b in df.columns:
            axes[0, 1].plot(epochs, df[col_v_dice_b], label='Val Benign (U Lành)', color='#1f77b4', linewidth=1.8, linestyle=':')
        if col_v_dice_m and col_v_dice_m in df.columns:
            axes[0, 1].plot(epochs, df[col_v_dice_m], label='Val Malignant (U Ác)', color='#ff7f0e', linewidth=1.8, linestyle='-.')
        axes[0, 1].set_title('B. Dice Score Progression', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_ylim(bottom=0.0, top=1.0)
        axes[0, 1].grid(True, linestyle=':', alpha=0.6)
        axes[0, 1].legend()

    # Ô (1, 0): IoU, Precision, Recall
    has_metrics = False
    if col_v_iou and col_v_iou in df.columns:
        axes[1, 0].plot(epochs, df[col_v_iou], label='Validation IoU', color='#9467bd', linewidth=2.0)
        has_metrics = True
    if col_v_prec and col_v_prec in df.columns:
        axes[1, 0].plot(epochs, df[col_v_prec], label='Validation Precision', color='#8c564b', linewidth=2.0, linestyle='--')
        has_metrics = True
    if col_v_rec and col_v_rec in df.columns:
        axes[1, 0].plot(epochs, df[col_v_rec], label='Validation Recall', color='#e377c2', linewidth=2.0, linestyle='-.')
        has_metrics = True

    if has_metrics:
        axes[1, 0].set_title('C. Validation IoU, Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(bottom=0.0, top=1.0)
        axes[1, 0].grid(True, linestyle=':', alpha=0.6)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'Metrics IoU/Prec/Rec not available in this log', ha='center', va='center')

    # Ô (1, 1): Learning Rate hoặc Zoom Best Epoch
    if col_lr and col_lr in df.columns:
        axes[1, 1].plot(epochs, df[col_lr], label='Learning Rate', color='#e377c2', linewidth=2.0)
        axes[1, 1].set_title('D. Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, linestyle=':', alpha=0.6)
        axes[1, 1].legend()
    elif col_v_dice:
        # Nếu không có cột LR, hiển thị tóm tắt text thống kê cao nhất
        axes[1, 1].axis('off')
        summary_box = (
            f"📊 PERFORMANCE HIGHLIGHTS\n"
            f"-----------------------------------------\n"
            f"• Best Epoch         : #{best_epoch}\n"
            f"• Peak Val Dice      : {best_dice:.4f} ({best_dice*100:.2f}%)\n"
        )
        if col_v_loss:
            best_loss = df[col_v_loss].iloc[best_epoch_idx]
            summary_box += f"• Val Loss at Best Ep : {best_loss:.4f}\n"
        if col_v_dice_b and col_v_dice_b in df.columns:
            summary_box += f"• Benign Dice at Best : {df[col_v_dice_b].iloc[best_epoch_idx]:.4f}\n"
        if col_v_dice_m and col_v_dice_m in df.columns:
            summary_box += f"• Malignant Dice at Best : {df[col_v_dice_m].iloc[best_epoch_idx]:.4f}\n"

        axes[1, 1].text(
            0.1, 0.4, summary_box, fontsize=12, family='monospace',
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f0f8ff", edgecolor="#4682b4", linewidth=2)
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dashboard_path = os.path.join(output_dir, f"{model_type}_training_dashboard.png")
    plt.savefig(dashboard_path, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    print(f"🌟 Đã lưu ảnh Dashboard 4 ô tổng hợp: {dashboard_path}\n")
    return dashboard_path


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    model_type = getattr(cfg.MODEL, "TYPE", "dual_decoder_resnet")
    checkpoint_dir = getattr(cfg.CALLBACKS.MODEL_CHECKPOINT, "PATH", "checkpoint")
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(cfg.WORK_DIR, checkpoint_dir)

    output_dir = getattr(cfg, "OUTPUT_DIR", None)
    if not output_dir:
        output_dir = os.path.join(cfg.WORK_DIR, "outputs", "training_plots")
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(cfg.WORK_DIR, output_dir)

    log_file = getattr(cfg, "LOG_FILE", None)
    if log_file and not os.path.isabs(log_file):
        log_file = os.path.join(cfg.WORK_DIR, log_file)

    plot_history(
        log_file=log_file,
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        show_plots=False
    )


if __name__ == "__main__":
    main()

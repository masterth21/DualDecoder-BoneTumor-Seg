BỐI CẢNH
Repo DualDecoder-BoneTumor-Seg, commit a765325 (TensorFlow / Keras 3, Hydra). Model: UNet3+ với backbone ResNet34 (ImageNet),
2 decoder (Region, Boundary) + Fusion + Refinement; M0 = 1 decoder. Ablation 17 kịch bản (M0, B0, P1–P5 × 3 attention).
Làm 3 phần theo thứ tự, mỗi mục một commit nhỏ. KHÔNG đổi công thức attention (NaLa đã đúng bài báo), loss, augmentation, LR.

==================================================================================================
PHẦN 1 — SỬA LỖI (review + build thử cả 17 cấu hình ở a765325)
==================================================================================================
LỖI 1 (nghiêm trọng) — Bottleneck attention không được nối vào decoder UNet3+
  models/dual_decoder_resnet.py: nhánh decoder_type == "unet3plus" gọi
      build_unet3plus_decoder([e1, e2, e3, e4, e5], ...)
  nên `bottleneck` (e5 sau NaLa/LogLinear/Multipole) bị bỏ, Keras cắt luôn khối này khỏi graph.
  Đo thực tế: P1 = B0 = 30.26M tham số, P4 = P2, P5 = P3 → các kịch bản có bottleneck vô nghĩa.
  Sửa: truyền [e1, e2, e3, e4, bottleneck] cho CẢ HAI nhánh (Region và Boundary).

LỖI 2 (nghiêm trọng) — LogLinearCore không chạy
  Mọi cấu hình có log_linear (P1–P5_log, cả bottleneck lẫn gate) lỗi khi build:
      TypeError: The two structures don't have the same nested structure (tf.while_loop + TensorArray)
  Sửa: bỏ tf.while_loop/TensorArray, dùng vòng lặp Python với số level TĨNH:
      L = ceil(log2(max(H, W))) tính từ static shape (H, W của e3/e4/e5 luôn biết trước: 48, 24, 12),
      nếu shape động thì dùng L = max_L cố định (mặc định 6) và pad lên 2^L.
  Đồng thời sửa sai lệch biên: avg-pool hiện tính cả các ô pad bằng 0 → trung bình bucket bị kéo nhỏ ở mép ảnh.
  Dùng tổng (sum-pool) cho KV và cho một mặt nạ đếm (1 ở ô thật, 0 ở ô pad), rồi
      S_ℓ = (sum_KV_ℓ − sum_KV_{ℓ−1}) / max(count_ℓ − count_{ℓ−1}, 1)
  (với ℓ = 0: S_0 = KV của chính vị trí đó). Giữ nguyên: λ = softplus(Dense(q_src)), L2-norm q/k, GroupNorm, W_o.

LỖI 3 (nghiêm trọng) — MultipoleCore lỗi shape ở bottleneck 12×12
  L = 3 với đầu vào 12×12: 12 → 6 → 3 → 1 (conv 2×2 stride 2 'valid' cắt mất hàng lẻ), khi đi lên
  U(1×1) = 2×2 nhưng cần cộng với 3×3 → InvalidArgumentError.
  Sửa: trong call, pad q_src và kv_src lên bội của 2^L (pad 0 ở phải/dưới) trước khi dựng kim tự tháp,
  cắt về (H, W) gốc ở cuối. Đồng thời giới hạn L hiệu dụng sao cho cạnh nhỏ nhất ở level thô nhất ≥ 1.
  Giữ nguyên: D, U, W_q/k/v dùng chung, cửa sổ w×w, cộng thường.

LỖI 4 (nghiêm trọng) — M0 (DUAL_DECODER=false) không train được
  train_dual_decoder.py vẫn compile với loss/metrics cho cả 3 đầu ra:
      ValueError: Expected keys ['refined_output'] in loss dict, but found ['region_output', 'boundary_output', 'refined_output']
  Sửa:
  - train_dual_decoder.py: lọc losses, loss_weights, metrics theo model.output_names
    (bỏ metric 'region_output' nếu không có đầu ra đó).
  - data_generators/data_generator.py: DualDecoderWrapper(base, output_names) chỉ trả target có trong output_names
    (M0 chỉ cần 'refined_output'; không tính boundary khi không cần).
  - Monitor của EarlyStopping / ModelCheckpoint / ReduceLROnPlateau: Keras 3 có thể đặt tên metric khác khi model
    chỉ có 1 đầu ra (vd. "val_dice_coef" thay vì "val_refined_output_dice_coef"). Sau khi compile, chạy
    model.evaluate trên 1 batch val, lấy danh sách key thật, chọn key chứa "dice_coef" của refined_output
    (hoặc duy nhất), in ra "[INFO] Monitor: <key>" và dùng key đó cho cả 3 callback và CompactProgressCallback.
  - callbacks/comprehensive_metrics_callback.py: đọc dự đoán theo tên (model 1 đầu ra trả tensor, không phải list).

LỖI 5 (nhỏ) — dọn dẹp
  - Xoá file/lớp attention cũ không còn dùng (nếu còn sót) để tránh nhầm khi viết bài.
  - README: sơ đồ đúng luồng: ResNet34 (ImageNet) → e1..e4 + bottleneck(e5 [+ attention chung]) →
    2 decoder UNet3+ full-scale (gate trên skip cùng tầng ở tầng 3–4) → Fusion → Refinement.

==================================================================================================
PHẦN 2 — ĐỘ DÀY BIÊN CẤU HÌNH ĐƯỢC
==================================================================================================
Hiện tại nhãn biên sinh on-the-fly từ mask (sau augmentation) trong DualDecoderWrapper bằng
utils/boundary_utils.extract_boundary_tf(mask) với kernel_size mặc định = 3 → biên dày 2 px (1 trong + 1 ngoài)
ở 384×384; create_combined_results.py vẽ biên GT bằng cv2 kernel 3×3 cố định (dòng ~198 và ~252).
Sửa:
- config.yaml thêm:
    BOUNDARY:
      KERNEL_SIZE: 5     # số lẻ; độ dày dải biên = KERNEL_SIZE − 1 pixel ở 384×384 (5 → 4 px)
      EVAL_TOL: 2        # dung sai (pixel) khi chấm Boundary F1
- DualDecoderWrapper(base, output_names, kernel_size): truyền cfg.BOUNDARY.KERNEL_SIZE vào extract_boundary_tf.
- create_combined_results.py: biên GT để HIỂN THỊ dùng cùng KERNEL_SIZE (không hard-code 3).
- In log lúc train: "[INFO] Boundary band: <K−1> px (kernel <K>)".

==================================================================================================
PHẦN 3 — HD95 VÀ ĐỘ CHÍNH XÁC BIÊN
==================================================================================================
3.1 Tạo utils/seg_metrics.py với ĐÚNG nội dung dưới đây (đã kiểm tra: HD95 khớp định nghĩa medpy; dịch mask 3 px
    → HD95 = 3.0; BF1@3px = 1.0). Mọi nơi tính HD95 / Boundary F1 đều import từ file này, không viết lại.

```python
"""
Chỉ số hình học cho phân đoạn: HD95 và Boundary F1 (BF score).
Đơn vị khoảng cách: pixel ở độ phân giải đánh giá (384×384).
"""
import numpy as np
from scipy import ndimage

_STRUCT = ndimage.generate_binary_structure(2, 1)


def contour(mask):
    """Viền 1 pixel (phía trong) của mask nhị phân."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    return mask & ~ndimage.binary_erosion(mask, structure=_STRUCT, border_value=0)


def _dist_to(mask):
    """Khoảng cách Euclid từ mọi pixel tới pixel True gần nhất của mask."""
    return ndimage.distance_transform_edt(~mask)


def hd95(pred, gt):
    """
    HD95 đối xứng (định nghĩa giống medpy.metric.hd95):
      percentile 95 của hợp hai tập khoảng cách viền pred→gt và gt→pred.
    Trả về (giá_trị, trạng_thái):
      ("ok")       cả hai không rỗng
      ("both_empty") cả hai rỗng → không tính (None)
      ("missed")   gt có, pred rỗng → None (đếm riêng)
      ("false_pos") gt rỗng, pred có → None (đếm riêng)
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not gt.any() and not pred.any():
        return None, "both_empty"
    if gt.any() and not pred.any():
        return None, "missed"
    if not gt.any() and pred.any():
        return None, "false_pos"
    cp, cg = contour(pred), contour(gt)
    d_pg = _dist_to(cg)[cp]
    d_gp = _dist_to(cp)[cg]
    return float(np.percentile(np.hstack([d_pg, d_gp]), 95)), "ok"


def boundary_f1(pred_boundary, gt_boundary, tol=2):
    """
    BF score: precision = tỉ lệ pixel biên dự đoán nằm trong tol pixel quanh biên GT,
              recall    = tỉ lệ pixel biên GT nằm trong tol pixel quanh biên dự đoán.
    Trả về dict {precision, recall, f1} (0..1) hoặc None nếu cả hai rỗng.
    """
    pb, gb = pred_boundary.astype(bool), gt_boundary.astype(bool)
    if not pb.any() and not gb.any():
        return None
    if not pb.any() or not gb.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    prec = float((_dist_to(gb)[pb] <= tol).mean())
    rec = float((_dist_to(pb)[gb] <= tol).mean())
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def boundary_dice(pred_boundary, gt_boundary):
    """Dice pixel-chính-xác giữa hai bản đồ biên (khắt khe, không có dung sai)."""
    pb, gb = pred_boundary.astype(bool), gt_boundary.astype(bool)
    s = pb.sum() + gb.sum()
    return None if s == 0 else float(2 * (pb & gb).sum() / s)


def summarize_hd95(values_and_status):
    """Gom kết quả hd95 của nhiều ảnh → mean, median, số ảnh mỗi trạng thái."""
    vals = [v for v, s in values_and_status if s == "ok"]
    out = {k: sum(1 for _, s in values_and_status if s == k) for k in ["ok", "missed", "false_pos", "both_empty"]}
    out["mean"] = float(np.mean(vals)) if vals else float("nan")
    out["median"] = float(np.median(vals)) if vals else float("nan")
    return out
```

3.2 Quy ước chung khi chấm (ở 384×384, sau argmax):
- HD95 tính cho từng ảnh, cho: lớp 1 (lành), lớp 2 (ác), và "u" nhị phân (lành ∪ ác).
  Báo: mean, median (chỉ trên ảnh trạng thái "ok") + số ảnh "missed" / "false_pos".
- Biên dự đoán: boundary_output → max theo các kênh u (kênh 1, 2) → ngưỡng 0.5.
  Biên GT để chấm: contour(mask_u) (viền 1 px của mask u nhị phân) — KHÔNG dùng dải dày, để kết quả không phụ thuộc
  KERNEL_SIZE. Chấm boundary_f1(biên_dự_đoán, contour(gt_u), tol=cfg.BOUNDARY.EVAL_TOL) và boundary_dice.
- Model M0 không có boundary_output → bỏ qua mục biên, ghi "N/A".

3.3 Báo cáo cuối khi train xong — callbacks/comprehensive_metrics_callback.py (on_train_end, trên tập VAL):
- Thêm vào bảng in ra và file CSV: HD95 (lành / ác / u) mean + median + số missed/false_pos;
  Boundary F1@tol, Boundary precision, recall (nếu có boundary_output).
- Bảng in ví dụ:
    Class / Metric | Dice | IoU | Precision | Recall | HD95 mean (px) | HD95 median (px)
    U Lanh         | ...                                   | ...            | ...
    U Ac           | ...
    U (nhị phân)   | ...
    Boundary F1@2px: xx.x% (P xx.x% | R xx.x%)   |   HD95 bỏ qua: missed n, false_pos m

3.4 Đánh giá test — create_combined_results.py và evaluate.py:
- prediction_details_test.csv thêm cột: hd95_benign, hd95_malignant, hd95_tumor, bf1, bf_precision, bf_recall, boundary_dice.
- summary_evaluation_test.txt và all_evaluations_history.csv thêm HD95 (mean/median từng lớp + u) và Boundary F1 trung bình.
- Ảnh ghép 5 cột:
    • Tiêu đề cột "Boundary Prediction":  "Boundary Prediction  BF1@{tol}px: 63.2%  (P 70.1% | R 57.5%)"
    • Tiêu đề cột "Pred Mask":            "Pred Mask  Dice: 0.7516 | HD95: 12.3 px"
    • Cột "Boundary Prediction" vẽ chồng: biên GT (contour) màu trắng mờ + biên dự đoán màu xanh, để nhìn ra chỗ lệch.
    • Ảnh không có u trong GT/dự đoán thì ghi "HD95: N/A".

KIỂM TRA (bắt buộc, Keras 3 giống Colab; báo bảng kết quả)
1. Build + forward 1 ảnh 384×384×3 ở float32 VÀ mixed_float16 cho 17 cấu hình
   (M0, B0, P1–P5 × {nalaformer, log_linear, multipole}; DECODER=unet3plus; WEIGHTS=imagenet).
   Bảng: tên | params | số đầu ra | số gate | có layer "*bottleneck*" | NaN.
   Yêu cầu: không lỗi; P1 > B0, P4 > P2, P5 > P3 về params; P*_log ≠ P*_mp.
2. Bottleneck 3 loại trên e5 (2,12,12,512): chạy được; lúc khởi tạo max|out − in| ≈ 0.
3. LogLinearCore: mọi λ bằng nhau, H = W = 16 → trùng linear attention toàn cục cùng chuẩn hoá (sai số < 1e-4);
   H = W = 12 (có pad) không còn sai lệch do ô pad.
4. MultipoleCore chạy với H, W ∈ {12, 24, 48, 50}; L = 0 và w ≥ H, W → trùng softmax toàn cục.
5. utils/seg_metrics.py: mask vuông 30×30 dịch 3 px → hd95 = 3.0; BF1@3px = 1.0; mask rỗng → trạng thái đúng.
6. Train thật 1 epoch (batch nhỏ) cho M0 và P5_log với KERNEL_SIZE=5: không lỗi; log in đúng "[INFO] Monitor: ..."
   và "[INFO] Boundary band: 4 px"; bảng cuối có HD95 (và Boundary F1 với P5_log, "N/A" với M0); ModelCheckpoint có lưu file.
7. create_combined_results.py với 3 ảnh: tiêu đề ảnh ghép có BF1 và HD95, CSV có đủ cột mới.
Commit cuối: "Fix bottleneck wiring, LogLinear/Multipole bugs, M0 path; add configurable boundary width, HD95 and Boundary F1".

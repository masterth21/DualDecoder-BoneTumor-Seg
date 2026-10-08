"""
Chỉ số hình học cho phân đoạn: HD95 và Boundary F1 (BF score).
Đơn vị khoảng cách: pixel ở độ phân giải đánh giá (384x384).
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
      percentile 95 của hợp hai tập khoảng cách viền pred->gt và gt->pred.
    Trả về (giá_trị, trạng_thái):
      ("ok")       cả hai không rỗng
      ("both_empty") cả hai rỗng -> không tính (None)
      ("missed")   gt có, pred rỗng -> None (đếm riêng)
      ("false_pos") gt rỗng, pred có -> None (đếm riêng)
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
    """Gom kết quả hd95 của nhiều ảnh -> mean, median, số ảnh mỗi trạng thái."""
    vals = [v for v, s in values_and_status if s == "ok"]
    out = {k: sum(1 for _, s in values_and_status if s == k) for k in ["ok", "missed", "false_pos", "both_empty"]}
    out["mean"] = float(np.mean(vals)) if vals else float("nan")
    out["median"] = float(np.median(vals)) if vals else float("nan")
    return out

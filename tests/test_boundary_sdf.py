"""
Tests: cờ MODEL.REFINEMENT (R2), BOUNDARY.MODE=binary, SDF head (u nhị phân).

Chạy:  python tests/test_boundary_sdf.py            (tất cả)
       python tests/test_boundary_sdf.py models     (chỉ bảng cấu hình a–e, h)
       python tests/test_boundary_sdf.py labels     (chỉ f, g, i)
Mỗi cấu hình model chạy trong 1 subprocess riêng (BACKBONE.WEIGHTS=none,
số tham số đếm ở 384x384, train step ở 128x128).
Biến môi trường: SDF_TEST_SIZE (mặc định 128), SDF_TEST_AMP_SIZE (mặc định = SDF_TEST_SIZE),
SDF_TEST_AMP = all | 0 | "B0_bin_sdf,..." (mặc định: all nếu có GPU, ngược lại chỉ B0_bin_sdf).
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

NONE_ATTN = {"TYPE": "none", "STAGES": []}

# name -> (overrides, expected params (M) ở 384 hoặc None)
CONFIGS = {
    "B0":               ({}, 30.26),
    "M0":               ({"MODEL.DUAL_DECODER": False}, 27.95),
    "R1":               ({"MODEL.FUSION_REFINEMENT": False}, 30.09),
    "R2":               ({"MODEL.REFINEMENT": False}, 30.14),
    "B0_bin":           ({"BOUNDARY.MODE": "binary"}, None),
    "B0_bin_sdf":       ({"BOUNDARY.MODE": "binary", "MODEL.SDF.ENABLE": True}, None),
    "R1_bin_sdf":       ({"MODEL.FUSION_REFINEMENT": False, "BOUNDARY.MODE": "binary", "MODEL.SDF.ENABLE": True}, None),
    "R2_bin_sdf":       ({"MODEL.REFINEMENT": False, "BOUNDARY.MODE": "binary", "MODEL.SDF.ENABLE": True}, None),
}


# =============================================================================
# Reference (bản sao ĐÓNG BĂNG của loss trước khi sửa, commit 6e04898)
# dùng để chứng minh loss B0 mặc định không đổi.
# =============================================================================
def _ref_losses():
    import tensorflow as tf

    def boundary_dice_loss(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
        class_weights = tf.constant([0.1, 0.45, 0.45], dtype=tf.float32)
        has_gt = tf.cast(tf.reduce_sum(y_true, axis=[1, 2]) > 0, tf.float32)
        sample_weights = class_weights * has_gt
        normalized_weights = sample_weights / (tf.reduce_sum(sample_weights, axis=-1, keepdims=True) + 1e-7)
        return tf.reduce_mean(tf.reduce_sum((1.0 - dice_per_class) * normalized_weights, axis=-1))

    def boundary_bce_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        class_weights = tf.constant([0.1, 0.45, 0.45], dtype=tf.float32)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce * class_weights)

    def boundary_bce_dice_loss(y_true, y_pred):
        return 0.5 * boundary_bce_loss(y_true, y_pred) + 0.5 * boundary_dice_loss(y_true, y_pred)

    def region_dice_ce_loss(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
        class_weights = tf.constant([0.1, 0.45, 0.45], dtype=tf.float32)
        has_gt = tf.cast(tf.reduce_sum(y_true, axis=[1, 2]) > 0, tf.float32)
        sample_weights = class_weights * has_gt
        normalized_weights = sample_weights / (tf.reduce_sum(sample_weights, axis=-1, keepdims=True) + 1e-7)
        loss_dice = tf.reduce_mean(tf.reduce_sum((1.0 - dice_per_class) * normalized_weights, axis=-1))
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss_ce = tf.reduce_mean(tf.reduce_sum(-y_true * tf.math.log(y_pred_clipped) * class_weights, axis=-1))
        return loss_dice + loss_ce

    return {'region_output': region_dice_ce_loss,
            'boundary_output': boundary_bce_dice_loss,
            'refined_output': region_dice_ce_loss}


# =============================================================================
# Helpers
# =============================================================================
def load_cfg(size, overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base=None):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    cfg.WORK_DIR = ROOT
    cfg.INPUT.HEIGHT = cfg.INPUT.WIDTH = size
    cfg.MODEL.BACKBONE.WEIGHTS = "none"
    cfg.MODEL.DECODER.TYPE = "unet3plus"
    cfg.MODEL.BOTTLENECK_ATTENTION = "none"
    cfg.MODEL.SKIP_ATTENTION.REGION = dict(NONE_ATTN)
    cfg.MODEL.SKIP_ATTENTION.BOUNDARY = dict(NONE_ATTN)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=False)
    return cfg


def synthetic_masks(batch, size):
    """Mask 3 lớp: mẫu 0 có u lành + u ác chạm nhau, mẫu 1 chỉ u ác, (mẫu 2+ không u)."""
    import numpy as np
    yy, xx = np.mgrid[:size, :size]
    m = np.zeros((batch, size, size), np.int32)
    c = size // 2; r = size // 5
    disk = (yy - c) ** 2 + (xx - c) ** 2 <= r ** 2
    m[0][disk & (xx < c)] = 1
    m[0][disk & (xx >= c)] = 2
    if batch > 1:
        m[1][(yy - c // 2) ** 2 + (xx - c // 2) ** 2 <= (r // 2) ** 2] = 2
    return np.eye(3, dtype=np.float32)[m]


def make_batch(size, batch=2):
    import numpy as np
    rng = np.random.RandomState(0)
    x = rng.rand(batch, size, size, 3).astype("float32")
    return x, synthetic_masks(batch, size)


def build_losses_from_cfg(cfg):
    from losses.dual_decoder_loss import get_dual_decoder_losses
    lw = cfg.LOSS.WEIGHTS
    sdf_node = cfg.MODEL.get("SDF", None)
    use_sdf = bool(sdf_node.get("ENABLE", False)) if sdf_node is not None else False
    w = {'region_output': float(lw.REGION), 'boundary_output': float(lw.BOUNDARY), 'refined_output': float(lw.REFINED)}
    if use_sdf:
        w['sdf_output'] = float(lw.get("SDF", 0.5))
    return get_dual_decoder_losses(
        w, boundary_mode=str(cfg.BOUNDARY.get("MODE", "class")),
        pos_weight=float(cfg.BOUNDARY.get("POS_WEIGHT", 1.0)),
        sdf_cfg=dict(cfg.SDF) if use_sdf else None)


def run_one(name, size, policy):
    """Chạy 1 cấu hình (trong subprocess)."""
    os.chdir(ROOT)
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from models.dual_decoder_resnet import build_dual_decoder_resnet
    from data_generators.data_generator import DualDecoderWrapper, dual_wrapper_kwargs

    overrides, expected = CONFIGS[name]
    tf.config.experimental.enable_op_determinism()

    # 1) params ở 384
    cfg384 = load_cfg(384, overrides)
    m384 = build_dual_decoder_resnet(cfg384)
    params = round(m384.count_params() / 1e6, 2)
    del m384
    tf.keras.backend.clear_session()

    # 2) train step ở `size`
    mixed_precision.set_global_policy(policy)
    cfg = load_cfg(size, overrides)
    tf.keras.utils.set_random_seed(42)
    model = build_dual_decoder_resnet(cfg)
    names = list(model.output_names)
    shapes = {n: list(o.shape[1:]) for n, o in zip(names, model.outputs)}
    dtypes = {n: str(o.dtype) for n, o in zip(names, model.outputs)}

    x, y = make_batch(size)

    class _Seq:
        def __len__(self): return 1
        def __getitem__(self, i): return x, y
    wrap = DualDecoderWrapper(_Seq(), output_names=names, **dual_wrapper_kwargs(cfg))
    _, targets = wrap[0]
    if not isinstance(targets, dict):
        targets = {names[0]: targets}
    tgt_shapes = {k: list(np.shape(v)[1:]) for k, v in targets.items()}

    losses, weights = build_losses_from_cfg(cfg)
    losses = {k: v for k, v in losses.items() if k in names}
    weights = {k: v for k, v in weights.items() if k in names}

    init_w = model.get_weights()

    def step(loss_dict):
        model.set_weights(init_w)
        opt = tf.keras.optimizers.Adam(1e-4, clipnorm=1.0)
        if policy == "mixed_float16":
            opt = mixed_precision.LossScaleOptimizer(opt)
        model.compile(optimizer=opt, loss=loss_dict, loss_weights=weights)
        t = targets if len(names) > 1 else targets[names[0]]
        out = model.train_on_batch(x, t, return_dict=True)
        return float(out["loss"])

    loss = step(losses)
    ref_loss = None
    if name == "B0":
        ref_loss = step(_ref_losses())

    return dict(name=name, params=params, expected=expected, outputs=names, shapes=shapes,
                dtypes=dtypes, target_shapes=tgt_shapes, loss=loss, ref_loss=ref_loss, policy=policy)


def check_one(d):
    errs = []
    if d["expected"] is not None and abs(d["params"] - d["expected"]) > 0.005:
        errs.append(f"params {d['params']} != {d['expected']}")
    if not (d["loss"] == d["loss"]) or abs(d["loss"]) == float("inf"):
        errs.append("loss không hữu hạn")
    if d["ref_loss"] is not None and abs(d["loss"] - d["ref_loss"]) > 1e-6:
        errs.append(f"loss B0 {d['loss']} != ref {d['ref_loss']}")
    n = d["name"]; sh = d["shapes"]
    if n == "M0" and d["outputs"] != ["refined_output"]:
        errs.append(f"outputs M0 {d['outputs']}")
    if n == "B0" and d["outputs"] != ["region_output", "boundary_output", "refined_output"]:
        errs.append(f"outputs B0 {d['outputs']}")
    if n.startswith("R1") and "refined_output" not in d["outputs"]:
        errs.append("R1 thiếu refined_output")
    if "boundary_output" in sh:
        exp_c = 1 if "bin" in n else 3
        if sh["boundary_output"][-1] != exp_c:
            errs.append(f"boundary_output {sh['boundary_output']} (kỳ vọng {exp_c} kênh)")
    if "sdf" in n:
        if sh.get("sdf_output", [None])[-1] != 1:
            errs.append(f"sdf_output thiếu/ sai shape {sh.get('sdf_output')}")
        if d["target_shapes"].get("sdf_output") != sh.get("sdf_output"):
            errs.append("target sdf shape không khớp")
    elif "sdf_output" in sh:
        errs.append("sdf_output xuất hiện khi chưa bật")
    for k, v in d["dtypes"].items():
        if "float32" not in v:
            errs.append(f"{k} dtype {v}")
    for k in d["outputs"]:
        if d["target_shapes"].get(k) != sh[k]:
            errs.append(f"target {k} {d['target_shapes'].get(k)} != output {sh[k]}")
    return errs


def run_models():
    size = int(os.environ.get("SDF_TEST_SIZE", 128))
    amp_size = int(os.environ.get("SDF_TEST_AMP_SIZE", size))
    # SDF_TEST_AMP: "all" | "0" | danh sách cấu hình, vd "B0_bin_sdf,R2_bin_sdf".
    # Mặc định: "all" nếu có GPU, ngược lại chỉ cấu hình d (fp16 trên CPU rất chậm).
    amp_env = os.environ.get("SDF_TEST_AMP")
    if amp_env is None:
        import tensorflow as tf
        amp_env = "all" if tf.config.list_physical_devices("GPU") else "B0_bin_sdf"
    amp_set = set(CONFIGS) if amp_env == "all" else (set() if amp_env == "0" else set(amp_env.split(",")))
    rows, bad = {}, 0
    for name in CONFIGS:
        policies = ["float32"] + (["mixed_float16"] if name in amp_set else [])
        for pol in policies:
            sz = amp_size if pol == "mixed_float16" else size
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", name, str(sz), pol],
                               capture_output=True, text=True, cwd=ROOT)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                bad += 1
                print(f"[FAIL] {name} ({pol}) ERROR\n{r.stderr[-2500:]}")
                continue
            d = json.loads(line[0][7:])
            errs = check_one(d)
            bad += bool(errs)
            rows.setdefault(name, {})[pol] = (d, errs)
            tag = "PASS" if not errs else "FAIL " + "; ".join(errs)
            extra = f" ref={d['ref_loss']:.6f}" if d["ref_loss"] is not None else ""
            print(f"[{tag}] {name:11} {pol:13} params={d['params']}M loss={d['loss']:.6f}{extra}", flush=True)

    print("\n| Cấu hình | Params (M) | Outputs | boundary_output | sdf_output | Loss 1 step (fp32) | Loss 1 step (AMP) | AMP OK |")
    print("|---|---|---|---|---|---|---|---|")
    for name, per in rows.items():
        d32 = per.get("float32", (None, ["missing"]))[0]
        d16, e16 = per.get("mixed_float16", (None, ["n/a"]))
        d = d32 or d16
        bsh = d["shapes"].get("boundary_output", "-")
        ssh = d["shapes"].get("sdf_output", "-")
        l32 = f"{d32['loss']:.4f}" if d32 else "-"
        l16 = f"{d16['loss']:.4f}" if d16 else "-"
        amp_ok = ("✓" if not e16 else "✗") if d16 else "n/a"
        print(f"| {name} | {d['params']} | {', '.join(d['outputs'])} | {bsh} | {ssh} | {l32} | {l16} | {amp_ok} |")
    return bad


# =============================================================================
# f. Nhãn SDF
# =============================================================================
def test_f_sdf_labels():
    import numpy as np
    from utils.boundary_utils import sdf_from_binary_np, sdf_from_onehot_np
    H = 128; T = 20.0
    yy, xx = np.mgrid[:H, :H]
    c, r = 64, 30
    rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    disk = rr <= r
    sdf = sdf_from_binary_np(disk, trunc=T)
    assert sdf.dtype == np.float32 and sdf.shape == (H, H)
    assert sdf.min() >= -1.0 and sdf.max() <= 1.0
    center = sdf[c, c]
    far = sdf[0, 0]
    edge_in = sdf[disk & (rr > r - 1)]          # pixel trong u sát biên
    edge_out = sdf[(~disk) & (rr < r + 1)]      # pixel ngoài u sát biên
    assert center <= -0.99, center              # tâm cách biên 30 px > TRUNC -> -1
    assert far == 1.0, far
    assert np.all(sdf[disk] < 0) and np.all(sdf[~disk] > 0), "sai dấu"
    assert np.abs(edge_in).max() <= 1.5 / T and np.abs(edge_out).max() <= 1.5 / T, "trên biên phải ≈ 0"
    # đơn điệu theo bán kính trên 1 tia trong vùng chưa cắt
    ray = sdf[c, c:c + r + 15]
    assert np.all(np.diff(ray) >= -1e-6), "SDF phải tăng dần từ tâm ra ngoài"
    # pixel cách biên 10px trong u ≈ -0.5
    assert abs(sdf[c, c + r - 10] - (-10.0 / T)) <= 1.0 / T
    empty = sdf_from_binary_np(np.zeros((H, H)), trunc=T)
    assert np.all(empty == 1.0)
    # batch one-hot: lành ∪ ác
    oh = synthetic_masks(3, H)
    s = sdf_from_onehot_np(oh, trunc=T)
    assert s.shape == (3, H, H, 1) and s.dtype == np.float32
    tumor = oh[..., 1:].max(-1)
    assert np.all((s[..., 0] < 0) == (tumor > 0.5))
    assert np.all(s[2] == 1.0)
    # không có "biên" giữa lành-ác trong SDF: dọc trục giữa của mẫu 0 vẫn âm sâu
    assert s[0, H // 2, H // 2, 0] <= -0.99
    print(f"  f. SDF: tâm={center:.3f}, xa ngoài={far:.3f}, biên∈[{min(edge_in.min(), edge_out.min()):.3f},"
          f"{max(edge_in.max(), edge_out.max()):.3f}], ảnh trống toàn +1, range [{sdf.min():.2f},{sdf.max():.2f}]")


# =============================================================================
# g. Nhãn biên binary vs class tại chỗ lành–ác tiếp giáp
# =============================================================================
def test_g_binary_boundary():
    import numpy as np
    from utils.boundary_utils import extract_boundary_tf, extract_boundary_binary_tf
    H = 128; k = 5
    oh = synthetic_masks(1, H)
    b_cls = extract_boundary_tf(oh, kernel_size=k).numpy()[0]       # (H,W,3)
    b_bin = extract_boundary_binary_tf(oh, kernel_size=k).numpy()[0]  # (H,W,1)
    assert b_bin.shape == (H, H, 1)
    c, r = H // 2, H // 5
    # vùng tiếp giáp lành–ác: cột x≈c, các hàng nằm sâu trong đĩa
    rows = slice(c - r // 2, c + r // 2)
    cols = slice(c - 2, c + 2)
    inter_cls = np.maximum(b_cls[..., 1], b_cls[..., 2])[rows, cols]
    inter_bin = b_bin[..., 0][rows, cols]
    assert inter_cls.sum() > 0, "mode class phải có viền ở chỗ lành–ác tiếp giáp"
    assert inter_bin.sum() == 0, "mode binary KHÔNG được có viền ở chỗ lành–ác tiếp giáp"
    # viền ngoài của u vẫn có
    assert b_bin[c, c - r, 0] == 1 and b_bin[c, c + r, 0] == 1
    # binary = biên của (lành ∪ ác)
    tumor = oh[..., 1:].max(-1, keepdims=True)
    ref = extract_boundary_tf(tumor, kernel_size=k).numpy()[0]
    assert np.array_equal(ref, b_bin)
    print(f"  g. Biên tại chỗ lành–ác: class={int(inter_cls.sum())} px, binary={int(inter_bin.sum())} px; "
          f"tổng px biên: class(max k1,k2)={int(np.maximum(b_cls[...,1], b_cls[...,2]).sum())}, binary={int(b_bin.sum())}")


# =============================================================================
# i. Data generator thật + DualDecoderWrapper, aug medium
# =============================================================================
def test_i_generator_medium():
    import numpy as np
    import cv2
    import tensorflow as tf
    from data_generators.tf_data_generator import DataGenerator
    from data_generators.data_generator import DualDecoderWrapper, dual_wrapper_kwargs
    from utils.boundary_utils import extract_boundary_binary_tf, sdf_from_onehot_np
    H = 128
    tmp = tempfile.mkdtemp()
    try:
        img_dir = os.path.join(tmp, "images"); msk_dir = os.path.join(tmp, "mask")
        os.makedirs(img_dir); os.makedirs(msk_dir)
        rng = np.random.RandomState(1)
        yy, xx = np.mgrid[:H, :H]
        for i in range(8):
            img = (rng.rand(H, H, 3) * 80 + 60).astype(np.uint8)
            m = np.zeros((H, H), np.uint8)
            if i % 4 != 3:
                cy, cx = rng.randint(40, 88, 2); r = rng.randint(14, 26)
                disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
                if i % 4 == 0:      # lành + ác chạm nhau
                    m[disk & (xx < cx)] = 128; m[disk & (xx >= cx)] = 255
                elif i % 4 == 1:
                    m[disk] = 128
                else:
                    m[disk] = 255
            cv2.imwrite(os.path.join(img_dir, f"img_{i}.png"), img)
            cv2.imwrite(os.path.join(msk_dir, f"img_{i}.png"), m)

        cfg = load_cfg(H, {"BOUNDARY.MODE": "binary", "MODEL.SDF.ENABLE": True})
        cfg.DATASET.TRAIN.IMAGES_PATH = img_dir; cfg.DATASET.TRAIN.MASK_PATH = msk_dir
        cfg.DATA_AUGMENTATION.ENABLE = True; cfg.DATA_AUGMENTATION.PRESET = "medium"
        cfg.HYPER_PARAMETERS.BATCH_SIZE = 2
        base = DataGenerator(cfg, mode="TRAIN")
        names = ["region_output", "boundary_output", "refined_output", "sdf_output"]
        kw = dual_wrapper_kwargs(cfg)
        assert kw["boundary_mode"] == "binary" and kw["sdf_trunc"] == 20.0
        gen = DualDecoderWrapper(base, output_names=names, **kw)
        printed = False
        for ep in range(2):
            for b in range(len(gen)):
                x, t = gen[b]
                if not printed:
                    print("  i. shapes:", {"x": tuple(x.shape)}, {k: tuple(np.shape(v)) for k, v in t.items()})
                    printed = True
                y = np.asarray(t["region_output"])
                assert np.array_equal(y, np.asarray(t["refined_output"]))
                assert set(np.unique(y)).issubset({0.0, 1.0}) and np.allclose(y.sum(-1), 1.0)
                bnd = np.asarray(t["boundary_output"]); sdf = np.asarray(t["sdf_output"])
                assert bnd.shape == y.shape[:3] + (1,) and sdf.shape == y.shape[:3] + (1,)
                assert sdf.dtype == np.float32 and sdf.min() >= -1 and sdf.max() <= 1
                # khớp với mask SAU augmentation
                assert np.array_equal(bnd, extract_boundary_binary_tf(y, kw["kernel_size"]).numpy())
                assert np.allclose(sdf, sdf_from_onehot_np(y, 20.0))
                tumor = y[..., 1:].max(-1)
                assert np.all((sdf[..., 0] < 0) == (tumor > 0.5)), "dấu SDF không khớp mask sau aug"
                # mọi pixel biên đều nằm gần đường đổi dấu SDF (|d| <= kernel/2 + 1 px)
                if bnd.sum() > 0:
                    assert np.abs(sdf[bnd > 0.5]).max() * 20.0 <= kw["kernel_size"] // 2 + 1.5
                for s in range(y.shape[0]):
                    if tumor[s].sum() == 0:
                        assert np.all(sdf[s] == 1.0) and bnd[s].sum() == 0
            gen.on_epoch_end()
        print("  i. generator + wrapper (aug medium, 2 epoch): biên & SDF khớp mask sau augmentation")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_labels():
    bad = 0
    for t in (test_f_sdf_labels, test_g_binary_boundary, test_i_generator_medium):
        try:
            t(); print(f"[PASS] {t.__name__}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[FAIL] {t.__name__}: {e}"); bad += 1
    return bad


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(run_one(sys.argv[2], int(sys.argv[3]), sys.argv[4])))
        sys.exit(0)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    bad = 0
    if what in ("all", "labels"):
        bad += run_labels()
    if what in ("all", "models"):
        bad += run_models()
    print("\nALL PASSED" if not bad else f"\n{bad} FAILED")
    sys.exit(1 if bad else 0)

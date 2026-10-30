"""
Tests: head phân loại cấp ảnh (MODEL.CLS_HEAD.ENABLE) — bình thường / u lành / u ác.

Chạy:  python tests/test_cls_head.py
Mỗi cấu hình model chạy trong 1 subprocess (BACKBONE.WEIGHTS=none, params đếm ở 384, train step ở 128).
"""
import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

CLS_ON = {"MODEL.CLS_HEAD.ENABLE": True}
BIN_SDF = {"BOUNDARY.MODE": "binary", "MODEL.SDF.ENABLE": True, "LOSS.WEIGHTS.SDF": 1.0}
# name -> (overrides, params kỳ vọng ở 384 hoặc None, outputs kỳ vọng)
CONFIGS = {
    "B0":             ({}, 30.26, ["region_output", "boundary_output", "refined_output"]),
    "M0":             ({"MODEL.DUAL_DECODER": False}, 27.95, ["refined_output"]),
    "B0_bin_sdf":     (dict(BIN_SDF), 30.25, ["region_output", "boundary_output", "refined_output", "sdf_output"]),
    # head phân loại thêm đúng 512*3 + 3 = 1,539 tham số: 30,254,865 + 1,539 = 30,256,404 (≈ 30.26M)
    "B0_bin_sdf_cls": (dict(BIN_SDF, **CLS_ON), 30.26, ["region_output", "boundary_output", "refined_output", "sdf_output", "cls_output"]),
    "M0_cls":         (dict({"MODEL.DUAL_DECODER": False}, **CLS_ON), 27.95, ["refined_output", "cls_output"]),
    "R1_cls":         (dict({"MODEL.FUSION_REFINEMENT": False}, **CLS_ON), 30.09, ["boundary_output", "refined_output", "cls_output"]),
}


def run_one(name, size, policy):
    os.chdir(ROOT)
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from test_boundary_sdf import load_cfg, make_batch, build_losses_from_cfg
    from models.dual_decoder_resnet import build_dual_decoder_resnet
    from data_generators.data_generator import DualDecoderWrapper, dual_wrapper_kwargs
    from losses.dual_decoder_loss import get_dual_decoder_losses

    ov, _, _ = CONFIGS[name]
    m384 = build_dual_decoder_resnet(load_cfg(384, ov))
    params = m384.count_params(); del m384
    tf.keras.backend.clear_session()

    mixed_precision.set_global_policy(policy)
    cfg = load_cfg(size, ov)
    tf.keras.utils.set_random_seed(42)
    model = build_dual_decoder_resnet(cfg)
    names = list(model.output_names)
    shapes = {n: list(o.shape[1:]) for n, o in zip(names, model.outputs)}
    dtypes = {n: str(o.dtype) for n, o in zip(names, model.outputs)}

    x, y = make_batch(size, batch=3)          # mẫu 0: lành + ác, mẫu 1: chỉ ác, mẫu 2: không u

    class _Seq:
        def __len__(self): return 1
        def __getitem__(self, i): return x, y
    _, t = DualDecoderWrapper(_Seq(), output_names=names, **dual_wrapper_kwargs(cfg))[0]
    if not isinstance(t, dict):
        t = {names[0]: t}

    losses, weights = build_losses_from_cfg(cfg)
    use_cls = "cls_output" in names
    if use_cls:
        cl, _ = get_dual_decoder_losses(None, cls_cfg={"CLASS_WEIGHTS": list(cfg.CLS.CLASS_WEIGHTS)})
        losses["cls_output"] = cl["cls_output"]; weights["cls_output"] = float(cfg.LOSS.WEIGHTS.CLS)
    losses = {k: v for k, v in losses.items() if k in names}
    weights = {k: v for k, v in weights.items() if k in names}
    opt = tf.keras.optimizers.Adam(1e-4, clipnorm=1.0)
    if policy == "mixed_float16":
        opt = mixed_precision.LossScaleOptimizer(opt)
    model.compile(optimizer=opt, loss=losses, loss_weights=weights)
    out = model.train_on_batch(x, t if len(names) > 1 else t[names[0]], return_dict=True)
    return dict(name=name, policy=policy, params=round(params / 1e6, 2), params_exact=int(params),
                outputs=names, shapes=shapes, dtypes=dtypes,
                cls_target=np.asarray(t["cls_output"]).tolist() if use_cls else None,
                loss=float(out["loss"]), cls_loss=float(out.get("cls_output_loss", float("nan"))) if use_cls else None)


def check(d):
    _, exp_p, exp_out = CONFIGS[d["name"]]
    errs = []
    if exp_p is not None and abs(d["params"] - exp_p) > 0.005:
        errs.append(f"params {d['params']} != {exp_p}")
    if d["outputs"] != exp_out:
        errs.append(f"outputs {d['outputs']} != {exp_out}")
    if not (abs(d["loss"]) < float("inf")):
        errs.append("loss không hữu hạn")
    if "cls_output" in d["outputs"]:
        if d["shapes"]["cls_output"] != [3]:
            errs.append(f"cls_output shape {d['shapes']['cls_output']}")
        if d["cls_target"] != [[0, 0, 1], [0, 0, 1], [1, 0, 0]] and d["cls_target"] != [[0, 1, 0], [0, 0, 1], [1, 0, 0]]:
            errs.append(f"nhãn cấp ảnh sai {d['cls_target']}")
        if not (d["cls_loss"] == d["cls_loss"]):
            errs.append("cls loss NaN")
    for k, v in d["dtypes"].items():
        if "float32" not in v:
            errs.append(f"{k} dtype {v}")
    return errs


# ----------------------------------------------------------------------------- test không cần model
def test_image_level_label():
    import numpy as np
    from data_generators.data_generator import image_level_onehot
    H = 32
    m = np.zeros((4, H, H), np.int32)
    m[1, :4, :4] = 1                         # chỉ lành
    m[2, :4, :4] = 2                         # chỉ ác
    m[3, :2, :2] = 1; m[3, 10:20, 10:20] = 2 # ác nhiều pixel hơn
    oh = np.eye(3, dtype=np.float32)[m]
    lab = image_level_onehot(oh).argmax(-1).tolist()
    assert lab == [0, 1, 2, 2], lab
    print(f"  nhãn cấp ảnh: {lab} (kỳ vọng [0, 1, 2, 2])")


def test_cls_loss():
    import numpy as np
    import tensorflow as tf
    from losses.dual_decoder_loss import make_cls_loss
    f = make_cls_loss([1.0, 1.0, 2.0])
    y = tf.constant([[1, 0, 0], [0, 0, 1]], tf.float32)
    perfect = float(f(y, y))
    p = tf.constant([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]], tf.float32)
    got = float(f(y, p))
    ref = (1 * -np.log(0.5) + 2 * -np.log(0.5)) / 3
    assert perfect < 1e-5 and abs(got - ref) < 1e-5, (perfect, got, ref)
    # trọng số: sai ở mẫu ác bị phạt nặng hơn sai ở mẫu bình thường
    a = float(f(y, tf.constant([[0.1, 0.45, 0.45], [0.0, 0.0, 1.0]])))
    b = float(f(y, tf.constant([[1.0, 0.0, 0.0], [0.45, 0.45, 0.1]])))
    assert b > a, (a, b)
    print(f"  cls loss: hoàn hảo {perfect:.2e}, ví dụ {got:.4f} = tham chiếu {ref:.4f}; sai ác {b:.3f} > sai bình thường {a:.3f}")


def test_default_config_off():
    from test_boundary_sdf import load_cfg
    cfg = load_cfg(384, {})
    assert bool(cfg.MODEL.CLS_HEAD.ENABLE) is False
    assert list(cfg.CLS.CLASS_WEIGHTS) == [1.0, 1.0, 2.0] and float(cfg.LOSS.WEIGHTS.CLS) == 0.5
    print("  config mặc định: CLS_HEAD tắt, CLASS_WEIGHTS [1, 1, 2], LOSS.WEIGHTS.CLS 0.5")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(run_one(sys.argv[2], int(sys.argv[3]), sys.argv[4]))); sys.exit(0)
    bad = 0
    for t in (test_image_level_label, test_cls_loss, test_default_config_off):
        try:
            t(); print(f"[PASS] {t.__name__}")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[FAIL] {t.__name__}: {e}"); bad += 1
    size = int(os.environ.get("CLS_TEST_SIZE", 128))
    amp = os.environ.get("CLS_TEST_AMP", "B0_bin_sdf_cls").split(",")
    for name in CONFIGS:
        for pol in ["float32"] + (["mixed_float16"] if name in amp else []):
            sz = int(os.environ.get("CLS_TEST_AMP_SIZE", size)) if pol == "mixed_float16" else size
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", name, str(sz), pol],
                               capture_output=True, text=True, cwd=ROOT)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                bad += 1; print(f"[FAIL] {name} ({pol})\n{r.stderr[-2500:]}"); continue
            d = json.loads(line[0][7:]); errs = check(d); bad += bool(errs)
            extra = f" cls_loss={d['cls_loss']:.4f}" if d["cls_loss"] is not None else ""
            print(f"[{'PASS' if not errs else 'FAIL ' + '; '.join(errs)}] {name:15} {pol:13} params={d['params_exact']:,} "
                  f"outputs={len(d['outputs'])} loss={d['loss']:.4f}{extra}", flush=True)
    print("\nALL PASSED" if not bad else f"\n{bad} FAILED")
    sys.exit(1 if bad else 0)

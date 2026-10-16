"""
apply_amp_fix.py — sửa lỗi AMP trong models/attention_cores.py.
Lỗi: các lớp con (DepthwiseConv2D, Dense, LayerNorm, GroupNorm, Conv2D, Conv2DTranspose) tạo trong build() của
NaLaCore / LogLinearCore / MultipoleCore KHÔNG kế thừa dtype của lớp cha. Trong gate (lớp cha float32) chúng vẫn chạy
float16 theo policy toàn cục → "Input 'y' of 'AddV2' Op has type float16 that does not match type float32".
Sửa: truyền dtype=self.dtype_policy cho mọi lớp con. Chạy từ thư mục gốc repo:  python apply_amp_fix.py
"""
import io, os, re, sys

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "attention_cores.py")
s = io.open(p, encoding="utf-8").read().replace("\r\n", "\n")
pat = re.compile(r"(= tf\.keras\.layers\.(?:Dense|DepthwiseConv2D|LayerNormalization|GroupNormalization|Conv2D|Conv2DTranspose)\()(.*)\)\s*$",
                 re.M)
n = 0
def fix(m):
    global n
    if "dtype=" in m.group(2):
        return m.group(0)
    n += 1
    return f"{m.group(1)}{m.group(2)}, dtype=self.dtype_policy)"
s2 = pat.sub(fix, s)
if n == 0 and "dtype=self.dtype_policy" in s:
    print("[skip] attention_cores.py đã sửa từ trước")
elif n != 20:
    sys.exit(f"[LỖI] mong đợi sửa 20 dòng nhưng tìm thấy {n} — gửi file attention_cores.py cho Claude")
else:
    io.open(p, "w", encoding="utf-8", newline="").write(s2)
    print(f"[ok] đã thêm dtype=self.dtype_policy cho {n} lớp con trong models/attention_cores.py")


# ---- thêm tests/test_amp.py ----
TEST = '"""AMP test: python tests/test_amp.py [names...]  (each config in its own subprocess)"""\nimport os, sys, json, subprocess\nROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nNONE = {"TYPE": "none", "STAGES": []}\ndef scen(name):\n    if name == "M0": return dict(dual=False, bott="none", reg=NONE, bnd=NONE)\n    if name == "B0": return dict(dual=True, bott="none", reg=NONE, bnd=NONE)\n    p, a = name.split("_", 1)\n    sk = {"TYPE": a, "STAGES": [3, 4]}\n    return {"P1": dict(dual=True, bott=a, reg=NONE, bnd=NONE),\n            "P2": dict(dual=True, bott="none", reg=sk, bnd=NONE),\n            "P3": dict(dual=True, bott="none", reg=sk, bnd=sk),\n            "P4": dict(dual=True, bott=a, reg=sk, bnd=NONE),\n            "P5": dict(dual=True, bott=a, reg=sk, bnd=sk)}[p]\nALL = ["M0", "B0"] + [f"P{i}_{a}" for i in range(1, 6) for a in ("nalaformer", "log_linear", "multipole")]\n\ndef run_one(name, size, policy):\n    sys.path.insert(0, ROOT); os.chdir(ROOT)\n    import numpy as np, tensorflow as tf\n    from tensorflow.keras import mixed_precision\n    from hydra import compose, initialize_config_dir\n    from omegaconf import OmegaConf\n    mixed_precision.set_global_policy(policy)\n    s = scen(name)\n    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base=None):\n        cfg = compose(config_name="config")\n    OmegaConf.set_struct(cfg, False)\n    cfg.INPUT.HEIGHT = cfg.INPUT.WIDTH = size\n    cfg.MODEL.DUAL_DECODER = s["dual"]; cfg.MODEL.DECODER.TYPE = "unet3plus"\n    cfg.MODEL.BACKBONE.WEIGHTS = "imagenet"  # untrained BN + x255 input overflows fp16 at inference\n    cfg.MODEL.BOTTLENECK_ATTENTION = s["bott"]\n    cfg.MODEL.SKIP_ATTENTION.REGION = s["reg"]; cfg.MODEL.SKIP_ATTENTION.BOUNDARY = s["bnd"]\n    from models.dual_decoder_resnet import build_dual_decoder_resnet\n    from models.attention_gates import ContextAttentionGate\n    model = build_dual_decoder_resnet(cfg)\n    gates = sum(isinstance(l, ContextAttentionGate) for l in model._flatten_layers())\n    x = np.random.rand(2, size, size, 3).astype("float32")\n    y = tf.one_hot(np.random.randint(0, 3, (2, size, size)), 3)\n    out = model.predict(x, verbose=0)\n    outs = out if isinstance(out, list) else [out]\n    nan = any(not np.isfinite(np.asarray(o)).all() for o in outs)\n    opt = tf.keras.optimizers.Adam(1e-4)\n    if policy == "mixed_float16": opt = mixed_precision.LossScaleOptimizer(opt)\n    names = model.output_names\n    model.compile(optimizer=opt, loss={n: "categorical_crossentropy" for n in names})\n    loss = model.train_on_batch(x, {n: y for n in names} if len(names) > 1 else y)\n    loss = float(np.ravel(loss)[0])\n    return dict(name=name, params=round(model.count_params() / 1e6, 2), gates=gates,\n                out_dtype=str(outs[-1].dtype), nan=bool(nan), loss=round(loss, 4),\n                ok=bool((not nan) and np.isfinite(loss)))\n\nif __name__ == "__main__":\n    if len(sys.argv) > 1 and sys.argv[1] == "--one":\n        print("RESULT " + json.dumps(run_one(sys.argv[2], int(sys.argv[3]), sys.argv[4]))); sys.exit(0)\n    names = sys.argv[1:] or ALL\n    size = int(os.environ.get("AMP_TEST_SIZE", 128)); policy = os.environ.get("AMP_TEST_POLICY", "mixed_float16")\n    print(f"{\'name\':16} {\'params(M)\':>9} {\'gates\':>5} {\'loss\':>8}  nan   ok")\n    bad = 0\n    for n in names:\n        r = subprocess.run([sys.executable, __file__, "--one", n, str(size), policy], capture_output=True, text=True)\n        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]\n        if not line:\n            bad += 1; print(f"{n:16} ERROR\\n" + r.stderr[-1500:]); continue\n        d = json.loads(line[0][7:]); bad += not d["ok"]\n        print(f"{n:16} {d[\'params\']:>9} {d[\'gates\']:>5} {d[\'loss\']:>8}  {d[\'nan\']!s:5} {d[\'ok\']}", flush=True)\n    sys.exit(1 if bad else 0)\n'
t = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "test_amp.py")
os.makedirs(os.path.dirname(t), exist_ok=True)
io.open(t, "w", encoding="utf-8", newline="\n").write(TEST)
print("[ok] đã tạo tests/test_amp.py  (chạy: python tests/test_amp.py)")

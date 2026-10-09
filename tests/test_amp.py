"""AMP test: python tests/test_amp.py [names...]  (each config in its own subprocess)"""
import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NONE = {"TYPE": "none", "STAGES": []}
def scen(name):
    if name == "M0": return dict(dual=False, bott="none", reg=NONE, bnd=NONE)
    if name == "B0": return dict(dual=True, bott="none", reg=NONE, bnd=NONE)
    p, a = name.split("_", 1)
    sk = {"TYPE": a, "STAGES": [3, 4]}
    return {"P1": dict(dual=True, bott=a, reg=NONE, bnd=NONE),
            "P2": dict(dual=True, bott="none", reg=sk, bnd=NONE),
            "P3": dict(dual=True, bott="none", reg=sk, bnd=sk),
            "P4": dict(dual=True, bott=a, reg=sk, bnd=NONE),
            "P5": dict(dual=True, bott=a, reg=sk, bnd=sk)}[p]
ALL = ["M0", "B0"] + [f"P{i}_{a}" for i in range(1, 6) for a in ("nalaformer", "log_linear", "multipole")]

def run_one(name, size, policy):
    sys.path.insert(0, ROOT); os.chdir(ROOT)
    import numpy as np, tensorflow as tf
    from tensorflow.keras import mixed_precision
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    mixed_precision.set_global_policy(policy)
    s = scen(name)
    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base=None):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    cfg.INPUT.HEIGHT = cfg.INPUT.WIDTH = size
    cfg.MODEL.DUAL_DECODER = s["dual"]; cfg.MODEL.DECODER.TYPE = "unet3plus"
    cfg.MODEL.BACKBONE.WEIGHTS = "imagenet"  # untrained BN + x255 input overflows fp16 at inference
    cfg.MODEL.BOTTLENECK_ATTENTION = s["bott"]
    cfg.MODEL.SKIP_ATTENTION.REGION = s["reg"]; cfg.MODEL.SKIP_ATTENTION.BOUNDARY = s["bnd"]
    from models.dual_decoder_resnet import build_dual_decoder_resnet
    from models.attention_gates import ContextAttentionGate
    model = build_dual_decoder_resnet(cfg)
    gates = sum(isinstance(l, ContextAttentionGate) for l in model._flatten_layers())
    x = np.random.rand(2, size, size, 3).astype("float32")
    y = tf.one_hot(np.random.randint(0, 3, (2, size, size)), 3)
    out = model.predict(x, verbose=0)
    outs = out if isinstance(out, list) else [out]
    nan = any(not np.isfinite(np.asarray(o)).all() for o in outs)
    opt = tf.keras.optimizers.Adam(1e-4)
    if policy == "mixed_float16": opt = mixed_precision.LossScaleOptimizer(opt)
    names = model.output_names
    model.compile(optimizer=opt, loss={n: "categorical_crossentropy" for n in names})
    loss = model.train_on_batch(x, {n: y for n in names} if len(names) > 1 else y)
    loss = float(np.ravel(loss)[0])
    return dict(name=name, params=round(model.count_params() / 1e6, 2), gates=gates,
                out_dtype=str(outs[-1].dtype), nan=bool(nan), loss=round(loss, 4),
                ok=bool((not nan) and np.isfinite(loss)))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(run_one(sys.argv[2], int(sys.argv[3]), sys.argv[4]))); sys.exit(0)
    names = sys.argv[1:] or ALL
    size = int(os.environ.get("AMP_TEST_SIZE", 128)); policy = os.environ.get("AMP_TEST_POLICY", "mixed_float16")
    print(f"{'name':16} {'params(M)':>9} {'gates':>5} {'loss':>8}  nan   ok")
    bad = 0
    for n in names:
        r = subprocess.run([sys.executable, __file__, "--one", n, str(size), policy], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            bad += 1; print(f"{n:16} ERROR\n" + r.stderr[-1500:]); continue
        d = json.loads(line[0][7:]); bad += not d["ok"]
        print(f"{n:16} {d['params']:>9} {d['gates']:>5} {d['loss']:>8}  {d['nan']!s:5} {d['ok']}", flush=True)
    sys.exit(1 if bad else 0)

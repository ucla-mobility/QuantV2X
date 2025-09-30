import os, glob, argparse, json
import numpy as np
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
def parse_shapes_arg(arg):
    if not arg:
        return {}
    out = {}
    for kv in arg.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if ":" not in kv:
            raise ValueError(f"Bad shapes token (missing ':'): {kv}")
        name, shape_str = kv.split(":", 1)
        dims = tuple(int(x) for x in shape_str.split("x") if x)
        out[name.strip()] = dims
    return out
def add_profile_from_dicts(builder, network, config, min_dict, opt_dict, max_dict):
    dynamic_needed = any(any(int(d) == -1 for d in network.get_input(i).shape)
                         for i in range(network.num_inputs))
    if not dynamic_needed:
        print("[INFO] No dynamic dimensions; no profile added.")
        return
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        net_shape = tuple(int(d) for d in inp.shape)
        def pick(name, fallback):
            d = (min_dict if name == "min" else opt_dict if name == "opt" else max_dict)
            return d.get(inp.name, fallback)
        fallback = tuple(1 if d == -1 else d for d in net_shape)
        mins = pick("min", fallback)
        opts = pick("opt", fallback)
        maxs = pick("max", fallback)
        if not (len(mins) == len(opts) == len(maxs) == len(net_shape)):
            raise ValueError(f"Profile rank mismatch for {inp.name}: "
                             f"net={net_shape}, min={mins}, opt={opts}, max={maxs}")
        print(f"[PROFILE] {inp.name}: min={mins} opt={opts} max={maxs}")
        profile.set_shape(inp.name, mins, opts, maxs)
    config.add_optimization_profile(profile)
    print("[INFO] Optimization profile added (from CLI shapes).")
def add_profile_from_npz(builder, network, config, npz_dir, scan=128):
    dynamic_needed = any(any(int(d) == -1 for d in network.get_input(i).shape)
                         for i in range(network.num_inputs))
    if not dynamic_needed:
        print("[INFO] No dynamic dimensions; no profile added.")
        return
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))[:scan]
    if not files:
        raise RuntimeError(f"No .npz files found in {npz_dir} to derive profile.")
    print(f"[PROFILE] Scanning {len(files)} NPZ files for observed shapes...")
    seen = {}
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        seen[name] = []
    for fp in files:
        with np.load(fp) as data:
            for i in range(network.num_inputs):
                name = network.get_input(i).name
                if name in data:
                    seen[name].append(tuple(int(x) for x in data[name].shape))
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        net_shape = tuple(int(d) for d in inp.shape)
        obs = seen.get(inp.name, [])
        if not obs:
            print(f"[WARN] No observations for input '{inp.name}'. Using 1s for dynamic dims.")
            mins = tuple(1 if d == -1 else d for d in net_shape)
            opts = tuple(1 if d == -1 else d for d in net_shape)
            maxs = tuple(1 if d == -1 else d for d in net_shape)
        else:
            obs_arr = np.array([list(s) for s in obs], dtype=np.int64)
            mins, opts, maxs = [], [], []
            for j, d in enumerate(net_shape):
                if d == -1:
                    col = obs_arr[:, j]
                    dmin = int(max(1, col.min()))
                    dopt = int(np.median(col))
                    dmax = int(col.max())
                    if dmax == dopt:
                        dmax = max(dmax, dopt + 1)
                    mins.append(dmin); opts.append(dopt); maxs.append(dmax)
                else:
                    mins.append(d); opts.append(d); maxs.append(d)
        print(f"[PROFILE] {inp.name}: min={tuple(mins)} opt={tuple(opts)} max={tuple(maxs)}")
        profile.set_shape(inp.name, tuple(mins), tuple(opts), tuple(maxs))
    config.add_optimization_profile(profile)
    print("[INFO] Optimization profile added (from NPZ).")
def build_fp32_engine(onnx_path, engine_path, workspace_gb=4.0,
                      minShapes=None, optShapes=None, maxShapes=None,
                      npz_dir=None, profile_scan=128):
    if not os.path.exists(onnx_path):
        cands = sorted(glob.glob("*.onnx"), key=os.path.getmtime)
        if not cands:
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        onnx_path = cands[-1]
        print(f"[INFO] Using latest ONNX in cwd: {onnx_path}")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    print(f"[INFO] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] ONNX parse failed:")
            for i in range(parser.num_errors):
                print(f"  {i}: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")
    print("[INFO] ONNX parsed successfully.")
    names = set()
    for i in range(network.num_outputs):
        n = network.get_output(i).name
        if n in names:
            print(f"[WARN] Duplicate output '{n}' detected; unmarking.")
            network.unmark_output(network.get_output(i))
        else:
            names.add(n)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"[INPUT] {t.name}: shape={tuple(int(d) for d in t.shape)} dtype={t.dtype}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    min_dict = parse_shapes_arg(minShapes) if minShapes else {}
    opt_dict = parse_shapes_arg(optShapes) if optShapes else {}
    max_dict = parse_shapes_arg(maxShapes) if maxShapes else {}
    if min_dict or opt_dict or max_dict:
        add_profile_from_dicts(builder, network, config, min_dict, opt_dict, max_dict)
    elif npz_dir:
        add_profile_from_npz(builder, network, config, npz_dir, scan=profile_scan)
    else:
        if any(any(int(d) == -1 for d in network.get_input(i).shape)
               for i in range(network.num_inputs)):
            print("[WARN] Dynamic shapes detected but no profile supplied; "
                  "adding a trivial profile with 1s for dynamic dims (may be too tight).")
            add_profile_from_dicts(builder, network, config, {}, {}, {})
    print("[BUILD] Building FP32 engine...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build returned None (FP32).")
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"[SUCCESS] Saved FP32 engine to: {engine_path}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a TensorRT FP32 engine from ONNX.")
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--engine", default="model_fp32.plan", help="Output engine path")
    ap.add_argument("--workspace-gb", type=float, default=4.0, help="Workspace size (GB)")
    ap.add_argument("--minShapes", type=str, default=None,
                    help='e.g. "pairwise_t_matrix:1x1x1x4x4,record_len:1"')
    ap.add_argument("--optShapes", type=str, default=None,
                    help='e.g. "pairwise_t_matrix:1x3x3x4x4,record_len:1"')
    ap.add_argument("--maxShapes", type=str, default=None,
                    help='e.g. "pairwise_t_matrix:1x8x8x4x4,record_len:1"')
    ap.add_argument("--npz_dir", type=str, default=None,
                    help="Folder of NPZ snapshots to derive a profile (keys must match ONNX inputs)")
    ap.add_argument("--profile-scan", type=int, default=128,
                    help="How many NPZ files to scan for shapes")
    args = ap.parse_args()
    base, ext = os.path.splitext(args.engine)
    out_path = base if base.endswith("_fp32") else f"{base}_fp32"
    if not ext:
        ext = ".plan"
    out_path = out_path + ext
    build_fp32_engine(
        onnx_path=args.onnx,
        engine_path=out_path,
        workspace_gb=args.workspace_gb,
        minShapes=args.minShapes,
        optShapes=args.optShapes,
        maxShapes=args.maxShapes,
        npz_dir=args.npz_dir,
        profile_scan=args.profile_scan
    )
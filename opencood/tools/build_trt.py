import os, glob, argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ---------- Utilities ----------
def np_dtype_of(trt_dtype):
    return {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF:  np.float16,
        trt.DataType.INT8:  np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT64: np.int64,
        trt.DataType.BOOL:  np.bool_,
    }.get(trt_dtype, np.float32)

def iter_npz_dir(folder, input_names):
    """Iterator for loading calibration data from NPZ files"""
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        raise RuntimeError(f"No .npz files in {folder}")
    
    print(f"[INFO] Found {len(files)} calibration files")
    
    for idx, fp in enumerate(files):
        with np.load(fp) as data:
            sample = {}
            for name in input_names:
                if name not in data:
                    raise KeyError(f"{fp} missing key '{name}'")
                arr = np.ascontiguousarray(data[name])
                sample[name] = arr
            
            # Debug first sample shapes
            if idx == 0:
                print(f"[DEBUG] First calibration sample shapes:")
                for name, arr in sample.items():
                    print(f"  {name}: {arr.shape} dtype={arr.dtype}")
            
            yield sample

# ---------- Fixed Data-driven Calibrator ----------
class DataCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Fixed calibrator that properly handles dynamic shapes and data types
    """
    def __init__(self, input_meta, cache_file, data_iter, batch_size=1):
        super().__init__()
        self.input_meta = input_meta   # {name: {"dtype": trt.DataType, "shape": shape}}
        self.cache_file = cache_file
        self.data_iter = data_iter
        self.batch_size = batch_size
        self.dev_ptrs = {}
        self.dev_sizes = {}
        self.current_batch = None
        self.batch_count = 0
        
        # Pre-load first batch to ensure we have data
        try:
            self.current_batch = next(self.data_iter)
            print(f"[CALIB] Loaded first calibration batch")
        except StopIteration:
            raise RuntimeError("No calibration data available")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """
        CRITICAL: This method must return pointers in the EXACT order of 'names'
        """
        if self.current_batch is None:
            return None
        
        # Use current batch
        sample = self.current_batch
        
        # Prepare for next iteration
        try:
            self.current_batch = next(self.data_iter)
        except StopIteration:
            self.current_batch = None
        
        ptrs = []
        for name in names:
            if name not in sample:
                print(f"[WARNING] Calibrator requested '{name}' but it's not in sample. Available: {list(sample.keys())}")
                # Create dummy data if needed
                if name in self.input_meta:
                    shape = self.input_meta[name].get("shape", (1,))
                    dtype = np_dtype_of(self.input_meta[name]["dtype"])
                    arr = np.zeros(shape, dtype=dtype)
                else:
                    raise KeyError(f"Unknown input '{name}'")
            else:
                arr = sample[name]
            
            # Convert to correct dtype
            want_dtype = np_dtype_of(self.input_meta[name]["dtype"])
            if arr.dtype != want_dtype:
                arr = arr.astype(want_dtype, copy=False)
            
            # Ensure contiguous
            arr = np.ascontiguousarray(arr)
            
            # Allocate device memory
            nbytes = arr.nbytes
            if name not in self.dev_ptrs or nbytes > self.dev_sizes.get(name, 0):
                if name in self.dev_ptrs:
                    # Free old allocation if it exists and is too small
                    try:
                        self.dev_ptrs[name].free()
                    except:
                        pass
                self.dev_ptrs[name] = cuda.mem_alloc(nbytes)
                self.dev_sizes[name] = nbytes
            
            # Copy to device
            cuda.memcpy_htod(self.dev_ptrs[name], arr)
            ptrs.append(int(self.dev_ptrs[name]))
        
        self.batch_count += 1
        if self.batch_count % 10 == 0:
            print(f"[CALIB] Processed {self.batch_count} calibration batches")
        
        return ptrs

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[CALIB] Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[CALIB] Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ---------- Fixed profile builder ----------
def add_profile_from_npz(builder, network, config, npz_dir, input_names, scan=128):
    """Build optimization profile with proper shape handling"""
    
    # Check for dynamic dimensions
    has_dynamic = False
    for i in range(network.num_inputs):
        shape = network.get_input(i).shape
        if any(int(d) == -1 for d in shape):
            has_dynamic = True
            break
    
    if not has_dynamic:
        print("[INFO] No dynamic dimensions found, skipping profile creation")
        return
    
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))[:scan]
    if not files:
        raise RuntimeError(f"No .npz files to derive profile in {npz_dir}")
    
    print(f"[PROFILE] Analyzing {len(files)} files for shape profile")
    
    # Collect observed shapes
    seen = {name: [] for name in input_names}
    for fp in files:
        with np.load(fp) as data:
            for name in input_names:
                if name in data:
                    seen[name].append(tuple(int(x) for x in data[name].shape))
    
    # Create profile
    profile = builder.create_optimization_profile()
    
    for i in range(network.num_inputs):
        t = network.get_input(i)
        net_shape = tuple(int(d) for d in t.shape)
        
        if t.name not in seen or not seen[t.name]:
            print(f"[WARNING] No observations for input '{t.name}', using defaults")
            # Use network shape as-is for non-dynamic, or reasonable defaults
            mins = [d if d != -1 else 1 for d in net_shape]
            opts = [d if d != -1 else 1 for d in net_shape]
            maxs = [d if d != -1 else 1 for d in net_shape]
        else:
            obs_shapes = seen[t.name]
            obs_array = np.array([list(s) for s in obs_shapes])
            
            mins, opts, maxs = [], [], []
            for j, d in enumerate(net_shape):
                if d == -1:
                    col = obs_array[:, j]
                    dmin = int(max(1, col.min()))
                    dopt = int(np.median(col))
                    dmax = int(col.max())
                    
                    # For batch dimension, allow flexibility
                    if j == 0 and t.name in ['pairwise_t_matrix', 'record_len']:
                        dmin = 1
                        dmax = max(dmax, 10)  # Allow up to 10 for batch
                    
                    mins.append(dmin)
                    opts.append(dopt)
                    maxs.append(dmax)
                else:
                    mins.append(d)
                    opts.append(d)
                    maxs.append(d)
        
        print(f"[PROFILE] {t.name}: min={mins}, opt={opts}, max={maxs}")
        profile.set_shape(t.name, tuple(mins), tuple(opts), tuple(maxs))
    
    config.add_optimization_profile(profile)
    print("[PROFILE] Optimization profile added successfully")

# ---------- Main build function ----------
def build_engine(
    onnx_path,
    engine_path,
    precision="int8",                # 'int8' | 'fp16' | 'fp32'
    calib_npz_dir=None,
    cache_path=None,
    fp16_fallback=True,
    workspace_gb=2,
    profile_scan=100
):
    """Build a TensorRT engine with selectable precision."""
    
    # Find ONNX file
    if not os.path.exists(onnx_path):
        cands = sorted(glob.glob("*.onnx"), key=os.path.getmtime)
        if not cands:
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        onnx_path = cands[-1]
        print(f"[INFO] Using latest ONNX in cwd: {onnx_path}")
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"[INFO] Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()
        if not parser.parse(onnx_data):
            print("[ERROR] ONNX parse failed:")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")
    
    print(f"[INFO] ONNX model parsed successfully")
    
    # Check for duplicate outputs (common issue)
    output_names = set()
    for i in range(network.num_outputs):
        name = network.get_output(i).name
        if name in output_names:
            print(f"[WARNING] Duplicate output '{name}' detected, removing duplicate")
            network.unmark_output(network.get_output(i))
        else:
            output_names.add(name)
    
    # Gather input metadata
    input_meta, input_names = {}, []
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = tuple(int(d) for d in t.shape)
        
        # Store both dtype and shape for calibrator
        input_meta[t.name] = {
            "dtype": t.dtype,
            "shape": [d if d != -1 else 1 for d in shape]  # Use 1 for dynamic dims
        }
        input_names.append(t.name)
        print(f"[INPUT] {t.name}: shape={shape}, dtype={t.dtype}")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)

    prec = precision.lower()
    if prec not in ("int8", "fp16", "fp32"):
        raise ValueError("precision must be one of: int8, fp16, fp32")
    
    # Precision flags
    if prec == "int8":
        print("[PRECISION] Building INT8 engine")
        config.set_flag(trt.BuilderFlag.INT8)
        if fp16_fallback and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 fallback enabled for layers that cannot run in INT8")
    elif prec == "fp16":
        print("[PRECISION] Building FP16 engine")
        if not builder.platform_has_fast_fp16:
            print("[WARNING] Platform does not report fast FP16 support; proceeding anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("[PRECISION] Building FP32 engine (no precision flags)")
        # no flags for FP32

    # Add optimization profile for dynamic shapes
    # For fp16/fp32 with dynamic shapes, we still want profiles
    if any(int(d) == -1 for i in range(network.num_inputs) for d in network.get_input(i).shape):
        derived_npz_source = calib_npz_dir if calib_npz_dir else "."
        add_profile_from_npz(builder, network, config, derived_npz_source, input_names, scan=profile_scan)
    else:
        print("[INFO] Static shapes detected; no optimization profile required")

    # Attach calibrator only for INT8
    if prec == "int8":
        if calib_npz_dir is None:
            raise ValueError("--npz_dir is required for INT8 calibration")
        if cache_path is None:
            raise ValueError("--cache path is required for INT8 calibration")
        print("[CALIB] Setting up INT8 calibrator")
        data_iter = iter_npz_dir(calib_npz_dir, input_names)
        calibrator = DataCalibrator(input_meta, cache_path, data_iter, batch_size=1)
        config.int8_calibrator = calibrator
    else:
        # Ensure INT8 is off
        try:
            config.clear_flag(trt.BuilderFlag.INT8)
        except Exception:
            pass
        config.int8_calibrator = None

    # Build engine
    print("[BUILD] Building engine...")
    try:
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            # If requested INT8 and it failed, try FP16 fallback if allowed
            if prec == "int8" and fp16_fallback:
                print("[FALLBACK] INT8 build failed; attempting FP16 fallback...")
                config.clear_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = None
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                engine_bytes = builder.build_serialized_network(network, config)
                if engine_bytes is None:
                    raise RuntimeError("Engine build failed even with FP16 fallback")
                else:
                    # Adjust filename to reflect FP16
                    if "_int8" in engine_path:
                        engine_path = engine_path.replace("_int8", "_fp16")
                    else:
                        root, ext = os.path.splitext(engine_path)
                        engine_path = f"{root}_fp16{ext}"
                    print("[FALLBACK] Built FP16 engine instead")
            else:
                raise RuntimeError("Engine build returned None")
        
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"[SUCCESS] Engine saved to: {engine_path}")
        if prec == "int8" and cache_path:
            print(f"[SUCCESS] Calibration cache saved to: {cache_path}")
        
    except Exception as e:
        print(f"[ERROR] Engine build failed with exception: {e}")
        raise

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a TensorRT engine (FP32/FP16/INT8).")
    parser.add_argument("--onnx", default="model.onnx", help="Path to ONNX model")
    parser.add_argument("--engine", default="model.plan", help="Output engine path (suffix will reflect precision if needed)")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="int8", help="Engine precision to build")
    parser.add_argument("--npz_dir", help="Path to folder with calibration .npz files (required for INT8)")
    parser.add_argument("--cache", help="Calibration cache path (required for INT8)")
    parser.add_argument("--no-fp16-fallback", action="store_true", help="Disable FP16 fallback (used for INT8 build failures)")
    parser.add_argument("--workspace-gb", type=float, default=2.0, help="Workspace (GB) for TensorRT builder")
    parser.add_argument("--profile-scan", type=int, default=100, help="Number of NPZ files to scan to derive dynamic shape profiles")

    args = parser.parse_args()

    # If filename doesn't include precision, append a helpful suffix
    out_path = args.engine
    base, ext = os.path.splitext(out_path)
    if args.precision == "int8" and "_int8" not in base:
        out_path = f"{base}_int8{ext}"
    elif args.precision == "fp16" and "_fp16" not in base:
        out_path = f"{base}_fp16{ext}"
    elif args.precision == "fp32" and "_fp32" not in base:
        out_path = f"{base}_fp32{ext}"

    # Validate INT8 inputs
    if args.precision == "int8":
        if not args.npz_dir:
            raise SystemExit("ERROR: --npz_dir is required for INT8")
        if not args.cache:
            raise SystemExit("ERROR: --cache is required for INT8")

    build_engine(
        onnx_path=args.onnx,
        engine_path=out_path,
        precision=args.precision,
        calib_npz_dir=args.npz_dir,
        cache_path=args.cache,
        fp16_fallback=not args.no_fp16_fallback,
        workspace_gb=args.workspace_gb,
        profile_scan=args.profile_scan
    )

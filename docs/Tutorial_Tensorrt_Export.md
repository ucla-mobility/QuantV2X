# Tutorial on Exporting PyTorch Model to TensorRT

These instructions walk through exporting a baseline model to ONNX and building both INT8 and FP32 TensorRT engines.

### Export model to ONNX and dump calibration samples

We use the script below to export the PyTorch model into ONNX format and generate calibration samples for quantization.

```python
python opencood/tools/inference_onnx_dump_calibration.py     --model_dir ${MODEL_DIR}     --dump_npz
```

Arguments Explanation:

- `model_dir` : path to the trained PyTorch checkpoint.  
- `--dump_npz` : saves calibration samples in `.npz` format for quantization.

---

### Build TensorRT INT8 engine

After exporting the ONNX model and calibration samples, build the INT8 TensorRT engine:

```python
python opencood/tools/build_trt_int8.py     --onnx ${ONNX_ENGINE}.onnx     --npz_dir ${CALIBRATION_NPZ}     --cache ${CACHE_FILE}.cache     --engine ${MODEL_INT8}.plan
```

Arguments Explanation:

- `onnx` : exported ONNX model file.  
- `npz_dir` : folder containing calibration `.npz` samples.  
- `cache` : calibration cache file to accelerate future builds.  
- `engine` : output TensorRT INT8 engine file (`.plan`).  

---

### Build TensorRT FP32 engine

For a full-precision engine without quantization, build the FP32 TensorRT engine:

```python
python opencood/tools/build_trt_fp32.py     --onnx ${ONNX_ENGINE}.onnx     --npz_dir ${CALIBRATION_NPZ}     --engine ${MODEL_FP32}.plan
```

Arguments Explanation:

- `onnx` : exported ONNX model file.  
- `npz_dir` : folder containing calibration `.npz` samples.  
- `engine` : output TensorRT FP32 engine file (`.plan`).  

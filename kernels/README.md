# MixDQ SDXL Quantization Example with CUDA Kernels

This codebase contains the end-to-end pipeline of quantized SDXL-turbo **with latency and memory savings** (same as the [MixDQ huggingface pipeline](https://huggingface.co/nics-efc/MixDQ)). We share the cuda kernels of quantized layers for practical hardware resource savings. 

# Installation

1. Clone submodule through git (CUTLASS)

cd into the MixDQ root path (not `./kernels`, where the `.git` folder locates)

```
cd ..
git submodule init
git submodule add https://github.com/NVIDIA/cutlass.git ./kernels/third_party/
```

2. Intsall requirements

```
pip install -r requirements.txt
```

3. Install MixDQ Extension

3.1 Install from PyPI

If you simply wish to use the precompiled kernel without modifying the CUDA kernel, we provide a precompiled wheel that can be installed using the following command:

```
pip install -i https://pypi.org/simple/ mixdq-extension
```

Note that for the NVIDIA A100 GPU, it is recommended to install the latest version (0.6) of the mixdq_extension for better acceleration (optimized tiling parameters). For desktop GPU like RTX3090/RTX4070, we recommend using the 0.5 version. 

```
# for Nvidia A100
pip install -i https://pypi.org/simple/ mixdq-extension==0.6

# for other GPUs (e.g., RTX3090)
pip install -i https://pypi.org/simple/ mixdq-extension==0.5
```

3.2 Install Locally

```
pip install -e .
```

- The codebase has a local folder `./mixdq_extension`, python may serach for this local folder instead of the installed package, which will raise `no module named mixdq_extension._C` error. linking the corresponding _C file within python site-package path (if installed via wheel, `$YOUR_CONDA_ENV_PATH/lib/python3.8/site-packages/mixdq_extension/_C.cpython-38-x86_64-linux-gnu.so ./`), or the _C file within local build folder (if installed locally, `ln -s ../build/lib.linux-x86_64-cpython-38/mixdq_extension/_C.cpython-38-x86_64-linux-gnu.so ./` ) to `./mixdq_extension` will resolve this. 

# Usage

1. convert the MixDQ algorithm simulation quant checkpoint (e.g., `MixDQ/logs/debug/`) to new format, the default output path is `./output/new_ckpt.pth`. 

```
python convert_ckpt.py --ckpt $PATH_TO_MIXDQ_CKPT/ckpt.pth
```

2. Generate the FP16 reference image, and Test the accelerated version is producing correct result. This will generate images named `./result.png` (FP) and `./result_00.png` (INT8). (note that the memory cost printed is the complete pipeline, not just the unet)

```
bash scripts/run_fp16_output_picture.sh
bash scripts/run_quantize_output_picture.sh
```


3. Comparison of Memory savings for diffusion U-Net, run:

```
bash scripts/run_memory_compare.sh
```

The output will be like:

```
----- conducting FP infernece -----
Static (weights) memory usage: 6 G 581 M 512 957 K 512 Bytes (6725.93505859375 MBs)
Dynamic (acts) memory usage: 1 G 86 M 0 82 K 0 Bytes (1110.080078125 MBs)
Peak (total) memory usage: 7 G 668 M 512 15 K 512 Bytes (7836.01513671875 MBs)

----- conducting quantized infernece -----
Static (weights) memory usage: 2 G 527 M 512 328 K 512 Bytes (2575.32080078125 MBs)
Dynamic (acts) memory usage: 0 G 55 M 0 793 K 0 Bytes (55.7744140625 MBs)
Peak (total) memory usage: 2 G 583 M 512 97 K 512 Bytes (2631.09521484375 MBs)
```

| Memory Cost  (MB) | Static (Weight) | Dynamic (Act) | Peak Memory | 
|-------------------|---------------|-----------------|-------------|
| FP16 version      | 4998          | 240.88          |    5239     |
| Quantized version | 2575          | 55.77           |    2631     |
| Savings           | 1.94x         | 4.36x           |    1.99x    |


4. Comparison of Latency speedup for diffusion unet. We use the [nsight system](https://developer.nvidia.com/nsight-systems) to measure the latency, please follow the [tutorial](https://developer.nvidia.com/nsight-systems/get-started) for installing it. The [cuda graph]() is adopted to reduce the calling overhead of the operator. 

```
./scripts/run_quantize_profile.sh --cuda_graph_only
```

The above command will generate files in `./nsys_logs` folder. 

![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20240822164940.png)

The mesaured latency speedup are presented below (compared with the pytorch baseline). 

| UNet Latency (ms) | RTX3090 | RT4080 | A100 | 
|-------------------|---------------|-----------------|---------|
| FP16 version      | 43.6          | 36.1            |    30.7     |
| Quantized version | 34.2          | 24.9            |    28.8     |
| Speedup           | 1.27x         | 1.45x           |    1.07x    |


## Acknowledgements

Thanks [@hgyhungry](https://github.com/hgyhungry) for the majority of the cuda kernel development. This project is inspired by the open-source framwork [torchao](https://github.com/pytorch/ao). 


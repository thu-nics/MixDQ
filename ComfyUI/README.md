# ComfyUI MixDQ Setup

## Environment requisites

- **0.** Create conda environment `conda create -n kernel python=3.10` (you only need the hardware cuda kernel environment for extention)
- **1.** Follow the instructions here: https://github.com/thu-nics/MixDQ/tree/master/kernels to set up basic environment for the moduel.

- **2.** Follow the instructions here: https://github.com/comfyanonymous/ComfyUI#installing for manuel install comfyui for Linux.

- **3.** Git Clone and put the `MixDQ`repository under `./ComfyUI/custom_nodes/`

- **4.** Putting the sdxl-turbo model under `/ComfyUI/models/checkpoints/sdxl-turbo`. The model can be downloaded by Diffuser as below:

```python
from diffusers import AutoPipelineForText2Image
import torch
model_id = "stabilityai/sdxl-turbo"


cache_dir = "/share/public/diffusion_quant/huggingface/hub/"


save_path = "~/ComfyUI/models/checkpoints/sdxl-turbo/" 

pipe = AutoPipelineForText2Image.from_pretrained(model_id, cache_dir = cache_dir,torch_dtype=torch.float16, variant="fp16")

pipe.save_pretrained(save_path)
```

- It is worth noticing that torch 2.2.1+cu118 with xformers 0.0.25 may cause error in Diffusers inference when tested on our computer. To solve this problem, we recommend to uninstall xformers or use `xformers==0.0.20`(Although it may display version incompatibility but it works). For more details, you can see https://github.com/huggingface/diffusers/issues/9889.

## Run

1. Run `python main.py --cuda-device 0` to run ComfyUI, as `--cuda-device` chooses which GPU to run with.
2. Open browser and go to `http://127.0.0.1:8188`. 
3. Load the workflow by `mixdq_workflow.json` or `mixdq.png` in `ComfyUI/custom_nodes/MixDQ/workflow`
4. To display the text, another custom node is needed: `pythongosssss/ComfyUI-Custom-Scripts`, you can git clone the repo in `ComfyUI/custom_nodes/`. It is recommended to git clone `ComfyUI-Manager`, with which, you can install other nodes easily.
5. The image will be generated in both quantilized and nonquantilized method, and storage size and latency will be shown.

## Components
![mod1.png](..%2Fworkflow%2Fmod1.png)

![org_gen.png](..%2Fworkflow%2Forg_gen.png)

![mixdqquant.png](..%2Fworkflow%2Fmixdqquant.png)

There are four nodes provided in this project.

- **Load Pipeline**: Load the model and transfer pipeline to other nodes.
- **MixdqQuant**: Prompt as input, use the pipeline and quant the model to W8A8 to generate images. The output image and memory usage will be displayed as output.
- **OrgGen**: Prompt as input, use the original model to generate images for comparison. The output image and memory usage will be displayed as output.
- **MixdqIntegral**: Prompt as input, integration of quantitative and primitive models for inference efficiency and display of results.

## Notes on Use
The memory footprint size calculation is based on the memory footprint on CUDA. And ComfyUI's caching mechanism makes the model stay in CUDA after completing a run in order to quickly generate a new graph after changing the prompt later. So the memory footprint shown for the first time is the real situation, and you can thus compare the quantized model compressing the UNet by 2+GB.

The first model loading and quantization process takes a little bit longer, after that you can change the prompt to use the cached quantized model to quickly generate a new picture.
## Results
![mixdq.png](..%2Fworkflow%2Fmixdq.png)

- Quantilized:
```
Static (weights) memory usage: 4 G 142 M 162 271 K 162 Bytes (4238.264802932739 MBs)
Dynamic (acts) memory usage:1 G 78 M 962 217 K 962 Bytes (1102.2128314971924 MBs)
Peak (total) memory usage:5 G 291 M 271 505 K 271 Bytes (5411.49342250824 MBs)
```

- Non:
```
Static (weights) memory usage: 6 G 480 M 166 419 K 166 Bytes (6624.4093379974365 MBs)
Dynamic (acts) memory usage:1 G 78 M 962 217 K 962 Bytes (1102.2128314971924 MBs)
Peak (total) memory usage:7 G 621 M 275 525 K 275 Bytes (7789.512957572937 MBs)
```


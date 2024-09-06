
# The Mixed Precision Search Process

* Note: we have already provided the final mixed precision configs in 'mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/', to conduct mixed precision search to obtain the final bit-width configuration, you could follow the below steps:

### Description of the whole process

> We Take w5a8 for an example

(1) **PTQ**: process is conducted for each candidate bit-width (2/4/8), the corresponding quant params are saved in `ckpt.pth`. 

(2) **Sensitivity Analysis**: is conducted to obtain a config file (e.g., `bs32_sqnr_weight.yaml`), containing the layer-wise sensitivity. 
    - The "metric-decouple" is conducted during the sensitivity analysis. We split the layer types into 2 groups (the "quality-related", and the "content-related"), and measure their sensitivity respectively. 
    - The sensitivity analysis scheme: measure the difference of certain metric (SQNR for quality-related, SSIM for content-related) with **certain layer quantized**. 

(3) **Integer Programming**: given the memory budget, **iteratively runnning such process** yield a family of mixed-precision configurations on the pareto frontier. 

(4) Similar configurations are tested with image generation/metric ranking to determine the final ones. 

#### Phase 1: PTQ

```shell
python scripts/ptq.py --config ./configs/stable-diffusion/sdxl-turbo.yaml --outdir <your path to save ckpt> --seed 42
```

#### Phase 2: Get Sensitivity

run `./get_sensitivity.sh` to get the sensitivity for weight/act of the content/quality related layers. The sensitivity is generated in `${OUTDIR}/sensitivity_{w/a}_{content/quality}.yaml`

#### Phase 3: Integer Programming

install the utility package for integer programing `pip install ortools`. 

run `./integer_program.sh` to assign the mixed-precision bit-width. given the averaged bitwidth for W and A. This process will generate a number of candidate bitwidths placed under `${OUTDIR}/{weight/act}_{avg_bitwidth}_{K}`. The `K` is a weighting coefficient that controls the bit-width assigned for the 2 layer groups (quality/content related layers). We scan through multiple `K` values to determine proper value for `K`. 

- Noted that **not every configurations acquired this way is the good configurations**, you may need to run the programming process for multiple times with different seeds and "target_bitwidth" to generate a number of candidate configurations.

#### Phase 4: Choose the optimal config

- Choose a final mixed-precision config from the candidates based on the metric value / visual quality. Run `./mixed_precision_infer.sh`, firstly, this will generate one single image for each weight configuration under the folder `opt.image_folder`. Then, the one with the least MSE error with FP generated image is chosen as the best config (placed under the `$OUTDIR/final_weight_mp.yaml`). Then, similar process is conducted for choosing the activation config **with the current optimal weight config**, and will generate `$OUTDIR/final_{weight/act}_mp.yaml`


#### Inference with mixed precision quantized model

- Infer with the generated mixed precision plan, run `./final_mixed_precision_infer.sh`, the images will be generated under the `${OUTDIR}/generated_images`



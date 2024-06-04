
# The Mixed Precision Search Process

* Note: we have already provided the final mixed precision configs in 'mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/', to conduct mixed precision search to obtain the final bit-width configuration, you could follow the below steps:

### Description of the whole process

> We Take w5a8 for an example

(1) **PTQ**: process is conducted for each candidate bit-width (2/4/8), the corresponding quant params are saved in `ckpt.pth`. 

(2) **Sensitivity Analysis**: is conducted to obtain a config file (e.g., `bs32_sqnr_weight.yaml`), containing the layer-wise sensitivity. 
    - The "metric-decouple" is conducted during the sensitivity analysis. We split the layer types into 2 groups (the "quality-related", and the "content-related"), and measure their sensitivity respectively. 
    - The sensitivity analysis scheme: measure the difference of certain metric (SQNR for quality-related, SSIM for content-related) with **certain layer quantized**. 

(3) **Integer Programming**: given the memory budget, it yields a family of mixed-precision configurations on the pareto frontier. 

(4) Similar configurations are tested with image generation/metric ranking to determine the final ones. 

#### Phase 1: PTQ

```shell
python scripts/ptq.py --config ./configs/stable-diffusion/sdxl-turbo.yaml --outdir <your path to save ckpt> --seed 42
```

#### Phase 2: Get Sensitivity

```shell

# get sensitivity of weight
# sensitivity based on ssim
python mixed_precision_scripts/get_sensitivity/sdxl_turbo/quant_content.py --config ./configs/stable-diffusion/sdxl-turbo.yaml --base_path <outdir> --ckpt <quantized_ckpt_path> --sensitivity_type weight --template_config utils_files/bs32_ssim_weight.yaml --reference_img ./utils_files/reference_imgs_fp16
# sensitivity based on sqnr
python mixed_precision_scripts/get_sensitivity/sdxl_turbo/quant_quality.py --config ./configs/stable-diffusion/sdxl-turbo.yaml --base_path <outdir> --ckpt <quantized_ckpt_path> --unet_input_path ./utils_files/bs32_input_sdxl_turbo.pt  --unet_output_path ./utils_files/bs32_output_sdxl_turbo.pt --model_id stabilityai/sdxl-turbo --sensitivity_type weight --template_config utils_files/bs32_sqnr_weight.yaml

# get sensitivity of act
# sensitivity based on ssim
python mixed_precision_scripts/get_sensitivity/sdxl_turbo/quant_content.py --config ./configs/stable-diffusion/sdxl-turbo.yaml --base_path <outdir> --ckpt <quantized_ckpt_path> --sensitivity_type act --template_config utils_files/bs32_ssim_weight.yaml --reference_img ./utils_files/reference_imgs_fp16
# sensitivity based on sqnr
python mixed_precision_scripts/get_sensitivity/sdxl_turbo/quant_quality.py --config ./configs/stable-diffusion/sdxl-turbo.yaml --base_path <outdir> --ckpt <quantized_ckpt_path> --unet_input_path ./utils_files/bs32_input_sdxl_turbo.pt  --unet_output_path ./utils_files/bs32_output_sdxl_turbo.pt --model_id stabilityai/sdxl-turbo --sensitivity_type act --template_config utils_files/bs32_sqnr_weight.yaml
```


#### Phase 3: Integer Programming

```shell
# mixed precision for weight
python ./mixed_precision_scripts/optimize/integer_programming.py --mixed_precision_type weight --sensitivity_ssim ./mixed_precision_scripts/sensitivity_log/sdxl_turbo/weight/ssim/bs32_split_ssim_weight/sensitivity.yaml --sensitivity_sqnr ./mixed_precision_scripts/sensitivity_log/sdxl_turbo/weight/sqnr/bs32_split_sqnr_weight/sensitivity.yaml --para_size_config ./mixed_precision_scripts/optimize/tensor_ratio/sdxl_turbo/weight_ratio_config.yaml --mixed_precision_config <your path to save the mixde-precision configs for weihgt> --target_bitwidth 5

# mixed precision for act
python ./mixed_precision_scripts/optimize/integer_programming.py --mixed_precision_type act --sensitivity_ssim ./mixed_precision_scripts/sensitivity_log/sdxl_turbo/act/ssim/bs32_split_ssim_act/sensitivity.yaml --sensitivity_sqnr ./mixed_precision_scripts/sensitivity_log/sdxl_turbo/act/sqnr/bs32_split_sqnr_act/sensitivity.yaml --para_size_config ./mixed_precision_scripts/optimize/tensor_ratio/sdxl_turbo/act_ratio_config.yaml --mixed_precision_config <your path to save the mixde-precision configs for act> --target_bitwidth 7.7

# Due to the presence of activations that maintain FP16, we set the average bit width of the remaining activations to 7.7 so that the average bit width of all activations is not greater than 8 bit
```

#### Phase 4: Choose the optimal config

* give a final config based on the metric value

```shell
# Infer with mixed-precision quantization configurations for weights, obtaining one image per configuration, and then assess each to select the optimal mixed-precision configuration.
python mixed_precision_scripts/quant_inference_mp.py --base_path <outdir> --config ./configs/stable-diffusion/sdxl-turbo.yaml --ckpt <quantized_ckpt_path> --use_weight_mp --dir_weight_mp <your path to save the mixde-precision configs for weight> --skip_quant_act

# Infer with mixed-precision quantization configurations for activations, obtaining one image per configuration, and then assess each to select the optimal mixed-precision configuration.
# we should choose a mixed-precision config for weight in advance.
python mixed_precision_scripts/quant_inference_mp.py --base_path <outdir> --config ./configs/stable-diffusion/sdxl-turbo.yaml --ckpt <quantized_ckpt_path> --config_weight_mp <The final mixed precision config for weight> --use_act_mp --dir_act_mp <your path to save the mixde-precision configs for act>


# NOTE: the final config will be saved at the "base_path" dir
```

#### Inference with mixed precision quantized model

```shell
# inference with quantized weight (W5A16/32)
python scripts/quant_txt2img.py --base_path <outdir> --config configs/stable-diffusion/sdxl-turbo.yaml --ckpt <quantized_ckpt_path> --use_weight_mp --config_weight_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight/weight_5.02.yaml --skip_quant_act

# inference with quantized weight and act (W5A8)
python scripts/quant_txt2img.py --base_path <outdir> --config configs/stable-diffusion/sdxl-turbo.yaml --ckpt <quantized_ckpt_path> --use_act_mp --config_weight_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight/weight_5.02.yaml --config_act_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_7.77.yaml --act_protect ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_sensitivie_a8_1%.pt
```

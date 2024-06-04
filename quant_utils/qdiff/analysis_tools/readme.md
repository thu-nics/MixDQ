# Quantization Analysis Tools

- `sdxl_pipeline.py`: customized SDXL pipeline that saves the input & output activation
	- noted that **within different folder, we have different version of sdxl_pipeline code.**

- run `./distribution/test_sdxl_turbo.py` to generate the unet input and output 
	- dump file at `../error_func/unet_in_out/bs32_input.output_sdxl_turbo`

- run  `./error_func/quant_error.py` with args to generate quantized activation file

```
python quant_error.py --base_path /share/public/diffusion_quant/text2img_diffusers_ptq/w8a8_test/ --unet_input_path ./unet_in_out/bs32_input_sdxl_turbo --unet_output_path ./unet_in_out/bs32_output_sdxl_turbo --analysis_target 'quant_error_unet_output'
```

it will generate `sensitivity_{analysis_target}.log` within the quant ckpt dir, used for `error_plot.ipynb`

run `error_plot.ipynb` cells to generate `error_plot/sensitivity_log/sensitive_layers_xxx.pt`


- use the `act_weight_distribution.ipynb` to probe the activation dist.
	- noted that some analysis code relies on `error_func/sentsitity_log/unet_out_error/sensitivityq_layers_xxx`

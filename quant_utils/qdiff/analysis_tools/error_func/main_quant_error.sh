# calc the quant_error for each layer
# dump the run.log used for error_plot.ipynb

python quant_error.py \
	--base_path /share/public/diffusion_quant/text2img_diffusers_ptq/w8a8_test/ \
	--unet_input_path ./unet_in_out/bs32_input_sdxl_turbo \
	--unet_output_path ./unet_in_out/bs32_output_sdxl_turbo \
	--analysis_target 'quant_error_act'

## SDXL_Turbo Quant Inference
BASE_PATH='./logs'
PATHS=('./debug_sdxl_turbo')  # a list of logs to generate images

for path in "${PATHS[@]}"; do
	CKPT_PATH=${BASE_PATH}/${path}
	echo "Prcocessing $CKPT_PATH"

	# Normal Quant Inference
	#CUDA_VISIBLE_DEVICES=$1 python scripts/quant_txt2img.py --base_path $CKPT_PATH --batch_size 2 --num_imgs 8 

	# Mixed Precision Quant Inference
	WEIGHT_MP_CFG="./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight/weight_8.00.yaml"  # [weight_5.02.yaml, weight_8.00.yaml]
	ACT_MP_CFG="./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_8.00.yaml"    # [act_7.77.yaml, act_8.00.yaml]
	ACT_PROTECT="./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_sensitivie_a8_1%.pt"

	CUDA_VISIBLE_DEVICES=$1 python scripts/quant_txt2img.py \
		--base_path $CKPT_PATH --batch_size 2 --num_imgs 8  --prompt "a vanilla and chocolate mixing icecream cone, ice background" \
		--config_weight_mp $WEIGHT_MP_CFG \
		--config_act_mp  $ACT_MP_CFG \
		--act_protect  $ACT_PROTECT \
		--fp16

	# FP generation for reference
	#CUDA_VISIBLE_DEVICES=$1 python scripts/txt2img.py \
		#--config ./configs/stable-diffusion/sdxl_turbo.yaml \
		#--base_path $CKPT_PATH --batch_size 2 --num_imgs 8  --prompt "a vanilla and chocolate mixing icecream cone, ice background" \
		#--fp16
	
done

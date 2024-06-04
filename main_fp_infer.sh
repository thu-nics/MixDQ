## SDXL_Turbo FP Inference
config_name='sdxl_turbo.yaml' # quant config, but only model names are used
BASE_PATH='./logs/debug_fp'   # save image path

CUDA_VISIBLE_DEVICES=$1 python scripts/txt2img.py \
		--config ./configs/stable-diffusion/$config_name \
		--base_path $BASE_PATH --batch_size 2 --num_imgs 8  --prompt  "a vanilla and chocolate mixing icecream cone, ice background" \
		--fp16

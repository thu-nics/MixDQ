#config_name='lcm_lora.yaml'
#config_name='sdxl'
config_name='sdxl_turbo'


# ----------- generate the calib data ------------
CUDA_VISIBLE_DEVICES=$1 python scripts/gen_calib_data.py --config ./configs/stable-diffusion/$config_name.yaml --save_image_path ./debug_imgs

# ------ conduct PTQ for multiple precision ------
CUDA_VISIBLE_DEVICES=$1 python scripts/ptq.py --config ./configs/stable-diffusion/${config_name}.yaml --outdir ./logs/debug_$config_name --seed 42

BASE_PATH="./logs"
LOG_PATH="debug_${config_name}"
CKPT_PATH=${BASE_PATH}/${LOG_PATH}
echo "Prcocessing $CKPT_PATH"

# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=$1 python scripts/quant_txt2img.py --base_path $CKPT_PATH --batch_size 2 --num_imgs 8


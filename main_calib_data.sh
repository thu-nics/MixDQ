# config_name='lcm_lora.yaml'
# config_name='sdxl.yaml'
config_name='sdxl_turbo.yaml'

CUDA_VISIBLE_DEVICES=$1 python scripts/gen_calib_data.py --config ./configs/stable-diffusion/$config_name --save_image_path ./debug_imgs


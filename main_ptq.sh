# Run PTQ (SDXL)
# note that calib_bs=64 may cause OOM for SDXL

#cfg_name="sdxl.yaml"
cfg_name="sdxl_turbo.yaml"
#cfg_name="lcm_lora.yaml"

CUDA_VISIBLE_DEVICES=$2 python scripts/ptq.py --config ./configs/stable-diffusion/${cfg_name} --outdir ./logs/$1 --seed 42


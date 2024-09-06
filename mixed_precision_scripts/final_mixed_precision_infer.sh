OUTDIR='./logs/debug_sdxl_turbo'

# Mixed Precision Quant Inference
WEIGHT_MP_CFG="${OUTDIR}/final_weight_mp.yaml"  
ACT_MP_CFG="${OUTDIR}/final_act_mp.yaml"
ACT_PROTECT="./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_sensitivie_a8_1%.pt"

CUDA_VISIBLE_DEVICES=$1 python scripts/quant_txt2img.py \
    --base_path $OUTDIR \
    --config_weight_mp $WEIGHT_MP_CFG \
    --config_act_mp  $ACT_MP_CFG \
    --act_protect  $ACT_PROTECT \
    --fp16
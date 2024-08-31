OUTDIR="./logs/debug_sdxl_turbo"
REFERENCE_IMG_DIR="${OUTDIR}/generated_images_fp"   # The FP generated images as reference for MSE error
OUTPUT_IMG_DIR_W="./mixed_precision_scripts/test/weight/imgs/"
OUTPUT_IMG_DIR_A="./mixed_precision_scripts/test/act/imgs/"

# Infer with mixed-precision quantization configurations for weights, obtaining one image per configuration, and then assess each to select the optimal mixed-precision configuration.

python mixed_precision_scripts/quant_inference_mp.py \
    --base_path $OUTDIR \
    --config ./configs/stable-diffusion/sdxl_turbo.yaml \
    --use_weight_mp \
    --dir_weight_mp "./mixed_precision_scripts/test/weight/"  \
    --image_folder $OUTPUT_IMG_DIR_W \
    --reference_img $REFERENCE_IMG_DIR \
    --skip_quant_act 
 
# Infer with mixed-precision quantization configurations for activations, obtaining one image per configuration, and then assess each to select the optimal mixed-precision configuration.

python mixed_precision_scripts/quant_inference_mp.py\
    --base_path $OUTDIR \
    --config ./configs/stable-diffusion/sdxl_turbo.yaml \
    --config_weight_mp $OUTDIR/final_weight_mp.yaml \
    --use_act_mp \
    --dir_act_mp "./mixed_precision_scripts/test/act/" \
    --image_folder $OUTPUT_IMG_DIR_A \
    --reference_img $REFERENCE_IMG_DIR \
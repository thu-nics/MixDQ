OUTDIR="./logs/debug_sdxl_turbo"
REFERENCE_IMG_DIR="${OUTDIR}/generated_images_fp"  # the FP images generated when calib data generation

# ------ Activation Sensitivity Analysis --------
# python mixed_precision_scripts/get_sensitivity/sdxl_turbo/quant_content.py  \
#     --config ./configs/stable-diffusion/sdxl_turbo.yaml \
#     --base_path ${OUTDIR} \
#     --sensitivity_type weight \
#     --reference_img ${REFERENCE_IMG_DIR}  

# # sensitivity based on sqnr
# python mixed_precision_scripts/get_sensitivity/sdxl_turbo/quant_quality.py \
#     --config ./configs/stable-diffusion/sdxl_turbo.yaml \
#     --base_path ${OUTDIR}  \
#     --sensitivity_type weight \

# # ------ Activation Sensitivity Analysis --------
python mixed_precision_scripts/get_sensitivity/sdxl_turbo/quant_content.py  \
    --config ./configs/stable-diffusion/sdxl_turbo.yaml \
    --base_path ${OUTDIR} \
    --sensitivity_type act \
    --reference_img ${REFERENCE_IMG_DIR}  

# # sensitivity based on sqnr
# python mixed_precision_scripts/get_sensitivity/sdxl_turbo/quant_quality.py \
#     --config ./configs/stable-diffusion/sdxl_turbo.yaml \
#     --base_path ${OUTDIR}  \
#     --sensitivity_type act \

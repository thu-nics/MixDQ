OUTDIR="./mixed_precision_scripts/test/"
# REFERENCE_IMG_DIR="${OUTDIR}/generated_images_fp" 

# mixed precision for weight
python ./mixed_precision_scripts/optimize/integer_programming.py \
    --mixed_precision_type weight \
    --sensitivity_ssim ./mixed_precision_scripts/sensitivity_log/sdxl_turbo/weight/ssim/bs32_split_ssim_weight/sensitivity.yaml \
    --sensitivity_sqnr ./mixed_precision_scripts/sensitivity_log/sdxl_turbo/weight/sqnr/bs32_split_sqnr_weight/sensitivity.yaml  \
    --para_size_config ./mixed_precision_scripts/optimize/tensor_ratio/sdxl_turbo/weight_ratio_config.yaml \
    --mixed_precision_config ${OUTDIR}/weight \
    --target_bitwidth 5

# mixed precision for act
python ./mixed_precision_scripts/optimize/integer_programming.py \
    --mixed_precision_type act \
    --sensitivity_ssim ./mixed_precision_scripts/sensitivity_log/sdxl_turbo/act/ssim/bs32_split_ssim_act/sensitivity.yaml \
    --sensitivity_sqnr ./mixed_precision_scripts/sensitivity_log/sdxl_turbo/act/sqnr/bs32_split_sqnr_act/sensitivity.yaml \
    --para_size_config ./mixed_precision_scripts/optimize/tensor_ratio/sdxl_turbo/act_ratio_config.yaml \
    --mixed_precision_config ${OUTDIR}/act/ \
     --target_bitwidth 7.7

# Due to the presence of activations that maintain FP16, we set the average bit width of the remaining activations to 7.7 so that the average bit width of all activations is not greater than 8 bit
# Noted that **not every configurations acquired this way is the good configurations**, you may need to run the programming process for multiple times with different seeds and "target_bitwidth" to generate a number of candidate configurations.
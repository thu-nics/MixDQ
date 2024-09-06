extra_args=${1:-""}
echo $extra_args

nsys profile -s none -c cudaProfilerApi python quantize_sdxl.py \
	--quantize \
	--w_config ./cfgs/weight/uniform_8.yaml \
	--a_config ./cfgs/act/act_8.00.yaml \
	--ckpt ./quant_para_wsym_fp16.pth \
	--profile --profile_tool=nsys $extra_args

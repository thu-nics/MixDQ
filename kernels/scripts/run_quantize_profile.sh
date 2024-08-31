extra_args=${1:-""}

nsys profile -s none -c cudaProfilerApi  --force-overwrite true -o ./nsys_logs/a100_fp \
	python quantize_sdxl.py \
	--profile \
	--profile_tool nsys \
	--batch_size 1 \
	${extra_args}

# INT8 inference
nsys profile -s none -c cudaProfilerApi  --force-overwrite true -o ./nsys_logs/a100_w8a8 \
	python quantize_sdxl.py \
	--quantize \
	--w_config ./cfgs/weight/uniform_8.yaml \
	--a_config ./cfgs/act/act_8.00.yaml \
	--ckpt ./output/new_ckpt.pth \
	--profile \
	--profile_tool nsys \
	--bos \
	--batch_size 1 \
	${extra_args}

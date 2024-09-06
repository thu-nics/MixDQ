extra_args=${1:-""}

echo '----- conducting FP infernece -----'
python quantize_sdxl.py

echo '----- conducting quantized infernece -----'
python quantize_sdxl.py \
	--quantize \
	--w_config ./cfgs/weight/uniform_8.yaml \
	--a_config ./cfgs/act/act_8.00.yaml \
	--ckpt ./output/new_ckpt.pth \

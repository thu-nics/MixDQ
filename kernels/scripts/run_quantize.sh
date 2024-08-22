extra_args=${1:-""}

while(true); do

	python quantize_sdxl.py 
		${extra_args}

	python quantize_sdxl.py \
		--quantize \
		--w_config ./cfgs/weight/uniform_8.yaml \
		--a_config ./cfgs/act/act_8.00.yaml \
		--ckpt ./quant_para_wsym_fp16.pth \
		--bos \
		${extra_args}
	
	sleep 1
done


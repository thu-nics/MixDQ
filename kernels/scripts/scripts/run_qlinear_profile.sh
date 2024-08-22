extra_args=${1:-""}

nsys profile -s none -c cudaProfilerApi  --force-overwrite true -o ./nsys_logs/quantize_linear_test \
	python quantize_linear_test.py

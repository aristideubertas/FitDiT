export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="0" \
accelerate launch --config_file configs/deepspeed_config.yaml --num_processes 1 --main_process_port 10080 train_fitdit_stage1.py --config configs/train.yaml  # > log_stage1.txt 2>&1 & 

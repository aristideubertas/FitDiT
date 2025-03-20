CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
accelerate launch --config_file configs/deepspeed_config.yaml --num_processes 8 --main_process_port 10080 train_fitdit_stage1.py --config configs/train.yaml  # > log_stage1.txt 2>&1 & 

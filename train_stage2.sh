CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
accelerate launch --config_file configs/deepspeed_config.yaml --num_processes 1 --main_process_port 10080 train_fitdit_stage2.py --config configs/train.yaml  # > log_stage2.txt 2>&1 & 

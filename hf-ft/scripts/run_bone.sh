BASE_MODEL="/home/rwkv/JL/model/llama2-7b"
OUTPUT_PATH="/home/rwkv/JL/out_model/bone-test"
DATA_PATH="/home/rwkv/JL/model/MetaMathQA"

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --include=localhost:0 bone.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --data_path $DATA_PATH \
    --dataset_field query response \
    --dataset_split "train[:10]"\
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --merge True \
    --use_bone True \
    --run_name "bone" \
    --gradient_checkpointing True
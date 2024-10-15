BASE_MODEL="/home/rwkv/JL/llama2-7b"
OUTPUT_PATH="/home/rwkv/JL/out_model/llama-meta-bone-row*-merge"
DATA_PATH="/home/rwkv/JL/model/MetaMathQA"
BONE_PATH="/home/rwkv/JL/out_model/llama-meta-bone-row*/model.safetensors"
# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
python merge.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT_PATH \
    --merge True \
    --use_bone True \
    --load_bone $BONE_PATH

HF_ENDPOINT="https://hf-mirror.com" lm_eval --model hf \
    --model_args pretrained=/home/rwkv/JL/out_model/Gemma-meta-2b \
    --tasks gsm8k \
    --device cuda:0 \
    --cache_requests true \

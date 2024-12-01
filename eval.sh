HF_ENDPOINT="https://hf-mirror.com" lm_eval --model hf \
    --model_args pretrained=/home/rwkv/JL/out_model/hfllama-math-bone \
    --tasks gsm8k \
    --device cuda:2 \
    --cache_requests true \

load_model='/home/rwkv/JL/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
proj_dir='/home/rwkv/JL/out_model/metapissa-1.6b'
data_file='/home/rwkv/JL/data/metamath_text_document'

n_layer=24
n_embd=2048

micro_bsz=32
epoch_save=1
epoch_steps=6171 #6171
ctx_len=512

pissa_config='{"pissa_load":"","pissa_init":"","pissa_r":32,"svd_niter":4,"pissa_parts":"att,ffn"}'


python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--data_type binidx --vocab_size 65536 \
--ctx_len $ctx_len --epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save --micro_bsz $micro_bsz \
--n_layer $n_layer --n_embd $n_embd \
--pre_ffn 0 --head_qk 0 --lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 2 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--dataload pad --loss_mask pad \
--peft pissa --pissa_config $pissa_config --train_parts [] \
--wandb finetune
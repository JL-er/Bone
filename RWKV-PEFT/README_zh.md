<h1 align="center"> <p>🦚 RWKV-PEFT</p></h1>

### 不同方法配置需求

Consider the memory requirements for training the following models with an 4090 24GB GPU with 64GB of CPU RAM.(--strategy deepspeed_stage_1 --ctx_len 1024 --micro_bsz 1 --lora_r 64)

|   Model         | Full Finetuning | lora/pissa  | Qlora/Qpissa | State tuning |
| --------- | ---- | ---- | ---- | ---- |
| RWKV6-1.6B | OOM GPU | 7.4GB GPU | 5.6GB GPU | 6.4GB GPU |
| RWKV6-3B | OOM GPU | 12.1GB GPU | 8.2GB GPU | 9.4GB GPU |
| RWKV6-7B | OOM GPU | 23.7GB GPU(bsz 8 OOM) | 14.9GB GPU(bsz 8 need 19.5GB) | 18.1GB GPU |

# 快速开始
按照必要依赖
```
pip install requirements.txt
```
参考scripts中的示例修改路径以及所需参数（数据准备详细参考RWKV官方教程）
```
sh scripts/run_lora.sh
```
# 具体使用
- peft
参数peft中包含多个方法，详细查看简介，选择所需的方法后要配置相应的config
例如：
```
--peft lora --lora_config $lora_config
```
- Quant
在使用peft或state tuning时可使用Quant量化权重以减少显存占用
```
--quant int8/nf4
```
- infctx
RWKV系列特有的训练方式（无限长度训练），可配合任意微调方法一起使用，防止训练数据过长导致显存爆炸
ctx_len为你想训练的长度（根据训练数据长度设置） 
chunk_ctx根据显存适当调整，chunk_ctx是从ctx_len切片得到所以保证chunk_ctx小于ctx_len
添加脚本参数如下：
```
--train_type infctx --chunk_ctx 512 --ctx_len 2048
```
- State tuning
RWKV特有的微调方法，训练开销极低
- dataload
支持不同的数据采样，默认使用get(RWKV-LM)这是一种随机采样，将所有数据视作一条数据根据ctx_len随机切片便于并行
pad、only都是为了从每条数据起始开始采样
pad末尾填充，例如ctx_len为1024而当前采样数据实际长度为1000则会在末尾填充下一条数据前24个token便于并行
only仅支持bsz=1的情况，为了ctx_len设置为采样最大长度，当前采样数据长度大于ctx_len的部分会被截断
```
--dataload pad
```
- loss_mask
支持对qa 问题部分以及pad末尾填充部分遮掩，防止模型根据题目背答案增强模型泛化能力
```
--loss_mask qa/pad
```

## LoRA

# Release
- infctx
- fla --fla
- State tuning
- Quant(QPissa,QLora) --quant int8/nf4
- Pissa
- Lisa
- Lora
- dataload(get、pad、only)
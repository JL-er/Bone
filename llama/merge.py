import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal
import logging
import os
import torch.nn as nn

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from boneLinear import BoneLinear,get_gmm_model,print_trainable_parameters,save_bone

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"
logger = logging.getLogger(__name__)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    # Lora or PiSSA setting
    use_lora : Optional[bool] = field(default=False)
    adapter_name_or_path: Optional[str] = field(default=None,metadata={"help": ("Pre-initialized PiSSA adapter path; when this is not None, the following arguments are ignored."),},)
    init_lora_weights: Literal[True, "pissa_niter_4"] = field(default=True,metadata={"help": ("True -> LoRA; `pissa` -> PiSSA"),},)
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    # Quantization setting
    bits: int = field(default=16,metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    # TrainingArguments
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    merge : Optional[bool] = field(default=False,metadata={"help": "Merge the PiSSA adapter to the residual model or LoRA to the base model"},)
    bf16: Optional[bool] = field(default=True)
    run_name: str= field(default='None', metadata={"help": "Path to the training data."})
    use_bone: Optional[bool] = field(default=False)
    load_bone: Optional[str] = field(default="")


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def build_model(script_args, checkpoint_dir):
    if not script_args.use_lora: assert script_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if script_args.bf16 else torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    if compute_dtype == torch.float32 and script_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('='*80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('='*80)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
    # Tokenizer

    ###if script_args.use_gmm:
    print(model)


    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    return model
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file

def merge_bone(trainer: transformers.Trainer, bone_path,output_dir):
    bone_dict = load_file(bone_path)
    #print(bone_model)
    state_dict = trainer.model.state_dict()
    #bone_dict = bone_model.state_dict()
    cpu_state_dict = {}
    for key, value in state_dict.items():
        #v = value.cpu()
        # if key.endswith('.weight') and 'pro'  in key:  ###old
        #     prefix = key[:-len('.weight')]
        #     bone_name = prefix + '.bone'
        #     b,r = bone_dict[bone_name].shape
        #     ww = rearrange(state_dict[key].to(torch.bfloat16), '(a r1) (b r2) -> b a r1 r2', r1 = r, r2 = r)@bone_dict[bone_name].reshape(b//r, r, r)+bone_dict[bone_name].reshape(b//r, r, r)
        #     state_dict[key] += rearrange(ww, 'b a r1 r2 ->(a r1) (b r2) ')
        # cpu_state_dict[key] = state_dict[key]
        # if key.endswith('.weight') and 'pro'  in key:  ###Bone-row
        #     prefix = key[:-len('.weight')]
        #     bone_name = prefix + '.bone'
        #     b,r,_ = bone_dict[bone_name].shape
        #     ww = rearrange(state_dict[key].to(torch.bfloat16), '(a r1) (b r2) -> a b r1 r2', r1 = r, r2 = r)@bone_dict[bone_name]+bone_dict[bone_name]
        #     state_dict[key] += rearrange(ww, 'a b r1 r2 ->(a r1) (b r2) ')
        if key.endswith('.weight') and 'pro'  in key:  ###Bone-col
            prefix = key[:-len('.weight')]
            bone_name = prefix + '.bone'
            b,r,_ = bone_dict[bone_name].shape
            ww = rearrange(state_dict[key].to(torch.bfloat16), '(a r1) (b r2) -> b a r1 r2', r1 = r, r2 = r)@bone_dict[bone_name]+bone_dict[bone_name]
            state_dict[key] += rearrange(ww, 'b a r1 r2 ->(a r1) (b r2) ')
        cpu_state_dict[key] = state_dict[key]
        print(key)
        
    trainer._save(output_dir, state_dict=cpu_state_dict)

def merge():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
        
    if script_args.local_rank == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    logger.info("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    logger.info("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)


    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
    
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)
        
    # raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)

    # if script_args.local_rank > 0: 
    #     torch.distributed.barrier()
        
    # train_dataset = raw_train_datasets.map(
    #     train_tokenize_function,
    #     batched=True,
    #     batch_size=3000,
    #     num_proc=32,
    #     remove_columns=raw_train_datasets.column_names,
    #     load_from_cache_file=True,
    #     desc="Running tokenizer on train dataset",
    #     fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    # )

        
    # if script_args.local_rank == 0:
    #     torch.distributed.barrier()
    #     print(model)
    #     #model.print_trainable_parameters()
    #     print_trainable_parameters(model)

    #     logger.info("Training dataset samples:", len(train_dataset))
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
    #         logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    data_module = None
    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args)

    merge_bone(trainer=trainer, bone_path=script_args.load_bone,output_dir=script_args.output_dir)
    #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
    

if __name__ == "__main__":
    merge()


import torch, math
import torch.nn as nn
from torch.nn import functional as F
import functools
from einops import rearrange
BONE_CONFIG = {
    "r": 64,
}

# class BoneLinear(nn.Module):  ###need reshape
#     def __init__(self, in_features: int, out_features: int, bias: bool):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty((out_features, in_features)))
#         assert bias == False, "Biased QuantLinear not supported"
#         self.r = BONE_CONFIG["r"]
#         self.bone = nn.Parameter(torch.zeros(out_features, self.r))
#         self.in_features = in_features
#         self.out_features = out_features

#     def forward(self, x):
#         w = rearrange(self.weight, '(a r1) (b r2) -> b a r1 r2', r1 = self.r, r2 = self.r)@self.bone.reshape(self.out_features//self.r, self.r, -1)+self.bone.reshape(self.out_features//self.r, self.r, -1)
#         w = rearrange(w, 'b a r1 r2 ->(a r1) (b r2) ')
#         return F.linear(x,self.weight+w)
    
# class BoneLinear(nn.Module): #Bone-row
#     def __init__(self, in_features: int, out_features: int, bias: bool):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty((out_features, in_features)))
#         assert bias == False, "Biased QuantLinear not supported"
#         self.r = BONE_CONFIG["r"]
#         self.bone = nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r))
#         self.in_features = in_features
#         self.out_features = out_features

#     def forward(self, x):
#         w = rearrange(self.weight, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.bone+self.bone
#         w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
#         return F.linear(x,self.weight+w)


# class BoneLinear(nn.Module):  #Bone-AB
#     def __init__(self, in_features: int, out_features: int, bias: bool):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty((out_features, in_features)))
#         assert bias == False, "Biased QuantLinear not supported"
#         self.r = BONE_CONFIG["r"]
#         self.bone_A = nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r))
#         self.bone_B = nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r))
#         self.in_features = in_features
#         self.out_features = out_features

#     def forward(self, x):
#         w = rearrange(self.weight, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.bone_A+self.bone_B
#         w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
#         return F.linear(x,self.weight+w)
    

class BoneLinear(nn.Module):#Bone-row
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.r = BONE_CONFIG["r"]
        self.bone = nn.Parameter(torch.zeros(out_features//self.r, self.r, self.r))
        
    def forward(self, x):
        def fn_bone(weight, b, x, r):
            w = rearrange(weight, '(a r1) (b r2) -> b a r1 r2', r1 = r, r2 = r)@b+b
            w = rearrange(w, 'b a r1 r2 ->(a r1) (b r2) ')
            return F.linear(x , weight+w)
        return torch_checkpoint(fn_bone, self.weight, self.bone, x, self.r, use_reentrant=False)
    

def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
        trainable_params, all_param = get_nb_trainable_parameters(model)

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )


def get_parent_target_module(model: torch.nn.Module, key: str):
    final_dot_index = key.rfind('.')
    if final_dot_index == -1:
        parent_key = None
        target_key = key
    else:
        parent_key = key[:final_dot_index]
        target_key = key[final_dot_index+1:]
    parent_module = model.get_submodule(parent_key) if parent_key is not None else model
    return parent_module, target_key
def get_gmm_model(model):
    for name, sub_module in model.named_modules():
        parent_module, target_key = get_parent_target_module(model, name)
        if isinstance(sub_module, nn.Linear):
            #if 'proj' in name and 'k_proj' not in name and 'v_proj' not in name:
            if 'proj' in name :
                gmm_linear = BoneLinear(sub_module.in_features, sub_module.out_features, False)
                gmm_linear.weight = sub_module.weight
                setattr(parent_module, target_key, gmm_linear)
    model.requires_grad_(False)
    for name, module in model.named_modules():
        for pname, param in module.named_parameters():
            if 'bone' in pname:
                param.requires_grad = True
    for name, module in model.named_modules():
        for pname, param in module.named_parameters():
                print(pname, param.requires_grad)
        break
    print_trainable_parameters(model)
    return model

import transformers
#def merge_bone(model):
def save_bone(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {}
    for key, value in state_dict.items():
        #v = value.cpu()
        if 'bone' in key :
            cpu_state_dict[key] = value.cpu()
    print(cpu_state_dict)
    del state_dict
    trainer._save(output_dir, state_dict=cpu_state_dict)
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()//1024**2} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()//1024**2} MB")


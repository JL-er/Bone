# BONE: BLOCK AFFINE TRANSFORMATION AS PARAMETER EFFICIENT FINE-TUNING METHODS FOR LARGE LANGUAGE MODELS
https://arxiv.org/pdf/2409.15371

## Introduction
Low-Rank Adaptation (LoRA) has achieved remarkable training results by freezing the original weights and training only low-rank matrices, establishing itself as the predominant fine-tuning method for LLMs. In pursuit of performance closer to full-parameter training, a series of LoRA variants have emerged, such as LoRA+, PISSA, Olora, and LoRA-GA. However, these improvements complicate the initial setup of model training and increase initialization time. More importantly, they overlook the internal interactions of the original weight information. To address these issues, we introduce a novel theory, ``Weight Guide'' aimed at continuously guiding trainable matrices through the original weights during training to enhance the utilization of weight information. Based on this theory, we designed a new PEFT technique called Bone (\textbf{B}l\textbf{o}ck Affi\textbf{ne}), which not only enhances the utilization of original weight information but also emphasizes the internal connections between weights, leading to faster convergence and better data fitting. Experimental comparisons across two different LLM architectures (LLaMA2, RWKV6) and various parameter scales demonstrate that the Bone structure can achieve rapid convergence and superior data fitting without the need for complex initialization. For example, when fine-tuning LLaMA2-7B on the MetaMathQA dataset and validating on GSM8k and math benchmarks, Bone achieved fine-tuning scores of 49.36 and 8.8, respectively, outperforming PISSA by 5.84\% and 1.96\%.
<p float="left">
  <img src="./assets/llama2-7b.png" width="45%" />
  <img src="./assets/train_step.png" width="45%" /> 
</p>

<p>
  <img src="./assets/score.png" />
</p>

## How to Run
### HF Model
"PEFT-Bone is currently being merged into the official PEFT repository. In the future, you will only need to run 'pip install peft'
```
git clone https://github.com/JL-er/PEFT-Bone.git
cd PEFT-Bone
pip install -e .
```
```
git clone https://github.com/JL-er/Bone.git
```
```
cd cd Bone/hf-ft
sh scripts/run_bone.sh
sh scripts/merge_bone.sh
```
### RWKV Model
```
git clone https://github.com/JL-er/RWKV-PEFT.git
```
You can check the script settings in the Bone/rwkv-ft file and replace them in the RWKV-PEFT/scripts directory.
```
cd RWKV-PEFT
pip install -r requirements.txt
sh scripts/run_bone.sh
sh scripts/merge_bone.sh
```

## Bone
```
class BoneLinear(nn.Module):#Bone-col
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False
        self.r = BONE_CONFIG["r"]
        self.bone = nn.Parameter(torch.zeros(out_features//self.r, self.r, self.r))
    
    def forward(self, x):
        w = rearrange(self.weight, '(a r1) (b r2) -> b a r1 r2', r1 = self.r, r2 = self.r)@self.bone+self.bone
        w = rearrange(w, 'b a r1 r2 ->(a r1) (b r2) ')
        return F.linear(x,self.weight+w)
```
## Flash-Bone
coming soon!!!


# Citation
If you find this repo useful, please consider citing our works:
```bib
@misc{kang2024boneblockaffinetransformation,
      title={Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models}, 
      author={Jiale Kang},
      year={2024},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}

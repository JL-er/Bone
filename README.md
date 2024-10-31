# BONE: BLOCK AFFINE TRANSFORMATION AS PARAMETER EFFICIENT FINE-TUNING METHODS FOR LARGE LANGUAGE MODELS
https://arxiv.org/pdf/2409.15371

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

# BONE: BLOCK AFFINE TRANSFORMATION AS PARAMETER EFFICIENT FINE-TUNING METHODS FOR LARGE LANGUAGE MODELS
https://arxiv.org/pdf/2409.15371
<p float="left">
  <img src="./assets/llama2-7b.png" width="45%" />
  <img src="./assets/bone-col.png" width="45%" /> 
</p>

<p>
  <img src="./assets/score.png" />
</p>

## How to Run
### HF Model
```
cd llama
sh scripts/run_bone.sh
sh scripts/merge_bone.sh
```
### RWKV Model
```
cd RWKV-PEFT
sh scripts/run_bone.sh
sh scripts/merge_bone.sh
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

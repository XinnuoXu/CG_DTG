# Tree_enc_dec

Setup the basic enviornment

```
conda create -n Plan python=3.6
conda activate Plan
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Install transformers and [GEM dataset](https://gem-benchmark.com/tutorials/modeling)

```
# GEM dataset
pip install git+https://github.com/huggingface/datasets.git
pip install rouge_score
pip install sentencepiece
pip install transformers
pip install nltk
```

Check examples in dataset (python code):
```
from datasets import load_dataset
DATASET_NAME = "e2e_nlg"
data = load_dataset("gem", DATASET_NAME)
print (data['train'][0])
```
DATASET_NAME = 'e2e_nlg', 'web_nlg_en', 'xsum'. Examples are can be found in this [GEM-huggingface page](https://huggingface.co/datasets/gem#data-splits).

# Tree_enc_dec

Setup the basic enviornment

```
conda create -n Plan python=3.6
conda activate Plan
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -U sentence-transformers
conda install -c conda-forge hdbscan
pip install neuralcoref --no-binary neuralcoref
```

Install transformers and [GEM dataset](https://gem-benchmark.com/tutorials/modeling)

```
# GEM dataset
pip install git+https://github.com/huggingface/datasets.git
pip install rouge_score
pip install sentencepiece
pip install transformers
pip install nltk
pip install pyrouge
pip install sklearn
pip install multiprocess

# get ROUGE-1.5.5
git clone git@github.com:andersjo/pyrouge.git
pyrouge_set_rouge_path /absolute/path/to/pyrouge/tools/ROUGE-1.5.5
cd pyrouge/tools/ROUGE-1.5.5/data/
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```

Check examples in dataset (python code):
```
from datasets import load_dataset
DATASET_NAME = "e2e_nlg"
data = load_dataset("gem", DATASET_NAME)
print (data['train'][0])
```
DATASET_NAME = 'e2e_nlg', 'web_nlg_en', 'xsum'. Examples are can be found in this [GEM-huggingface page](https://huggingface.co/datasets/gem#data-splits).

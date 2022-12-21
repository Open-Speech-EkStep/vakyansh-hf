# Vakyansh models in HuggingFace format

## Installation 
```
pip install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
```

## Usage 
```{python}
from infer import Wav2vecHF
model = Wav2vecHF(<path to models>)
model.transcribe(<path to wav file>)
```
## Covert Vakyansh fairseq models to Huggingface format

```
cd wav2vec2_to_hf
```
Change the parameters in `parameters.py` appropriately:
```
HF_CONFIG = 'hf_config.json' #already present in repo
FINETUNED_CHECKPOINT = 'hindi_fairseq/him_4200.pt' # finetune checkpoint
PRETRAINED_CHECKPOINT = 'hindi_fairseq/CLSRIL-23.pt' 
DICT = 'hindi_fairseq/dict.ltr.txt'
LM = 'hindi_fairseq/lm.binary'
LEXICON = 'hindi_fairseq/lexicon.lst'
HF_DIRECTORY_PATH = 'hindi_hf'
```
```
python convert_checkpoint.py
python integrate_lm.py
```


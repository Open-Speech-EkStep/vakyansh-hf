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

from transformers import AutoProcessor
import parameters
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM

processor = AutoProcessor.from_pretrained(parameters.HF_DIRECTORY_PATH)
vocab_dict = processor.tokenizer.get_vocab()

sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=parameters.LM,
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

processor_with_lm.save_pretrained(parameters.HF_DIRECTORY_PATH)

with open(parameters.LEXICON, 'r', encoding='utf-8') as f:
    lexicon = f.read().splitlines()
    
words = [l.split('\t')[0] for l in lexicon]

with open(parameters.HF_DIRECTORY_PATH + '/language_model/unigrams.txt', 'w+') as f:
    for item in words:
        f.write("%s\n" % item)
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import glob
import soundfile as sf
from tqdm import tqdm


class Wav2vecHF:
    def __init__(self, model_path, lm):
        self.model_path = model_path
        self.lm = lm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = self.load_model()

    def load_model(self):
        model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
        if self.lm == 'viterbi':
            processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        elif self.lm == 'kenlm':
            processor = Wav2Vec2ProcessorWithLM.from_pretrained(self.model_path)
        else:
            print('Chose either viterbi or kenlm for decoding')
            exit()

        return processor, model.to(self.device)

    def transcribe(self, wav_path):                
        
        audio_input, sample_rate = sf.read(wav_path)

        inputs = self.processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = self.model(**inputs.to(self.device)).logits
        
        if self.lm == 'viterbi':
            text = self.processor.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
        elif self.lm == 'kenlm':
            result = self.processor.batch_decode(logits.cpu().numpy())
            text = result.text
        
        return text

    def transcribe_dir(self, audio_dir):
        wav_files = glob.glob(audio_dir + '/*.wav')
        for wav_file in tqdm(wav_files):
            text = self.transcribe(wav_file)
            filename = wav_file.split('/')[-1].replace('.wav', '.txt')
            with open(audio_dir + '/' + filename, mode='w+', encoding='utf-8') as ofile:
                ofile.write(text)

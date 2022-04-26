from infer import Wav2vecHF
import numpy as np
import soundfile as sf
import srt.timestamp_generator as tg

model_lm = Wav2vecHF('wav2vec2_to_hf/eng_hf', 'kenlm')
model_v = Wav2vecHF('wav2vec2_to_hf/eng_hf', 'viterbi')


def wav_to_pcm16(wav):
    ints = (wav * 32768).astype(np.int16)
    little_endian = ints.astype('<u2')
    wav_bytes = little_endian.tobytes()
    return wav_bytes


if __name__ == '__main__':
    audio, sr = sf.read('/Users/anirudhgupta/ekstepspeechrecognition/vakyansh-hf/market_mantra/market_mantra_24_april_enhanced.wav')
    wav_bytes = wav_to_pcm16(audio)
    start_times, end_times = tg.extract_time_stamps(wav_bytes)
    print(start_times)

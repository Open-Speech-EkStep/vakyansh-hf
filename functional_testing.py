from infer import Wav2vecHF
import gradio as gr
import nnresample

model_lm = Wav2vecHF('wav2vec2_to_hf/hindi_hf', 'kenlm')
model_v = Wav2vecHF('wav2vec2_to_hf/hindi_hf', 'viterbi')


def asr(wav_file, hot_words):
    if len(wav_file[1].shape) == 2:
        signal = wav_file[1].mean(-1)
    else:
        signal = wav_file[1]
    resampled_signal = nnresample.resample(signal, 16000, 48000)
    kenlm = model_lm.transcribe(resampled_signal, mode='numpy')
    viterbi = model_v.transcribe(resampled_signal, mode='numpy')
    hotwords_output = model_lm.transcribe(resampled_signal, [hot_words], mode='numpy')
    return kenlm, viterbi, hotwords_output


outputs = [gr.outputs.Textbox(label="KenLM"), gr.outputs.Textbox(label="Viterbi"),
           gr.outputs.Textbox(label="With hot-words")]
inputs = [gr.inputs.Audio("microphone", type='numpy'), "text"]
gr.Interface(asr, inputs, outputs, title="Tesling LM and hot words").launch(share=False)

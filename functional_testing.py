from infer import Wav2vecHF

model_lm = Wav2vecHF('/home/anirudh/ekstep-speech-recognition/vakyansh-wav2vec2-experimentation/checkpoints/hf','kenlm')
model_v = Wav2vecHF('/home/anirudh/ekstep-speech-recognition/vakyansh-wav2vec2-experimentation/checkpoints/hf','viterbi')

print(model_lm.transcribe('blindtest_442338.wav'))
print(model_v.transcribe('blindtest_442338.wav'))
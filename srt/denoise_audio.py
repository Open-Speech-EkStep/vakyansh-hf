import os


def enhance_audio(audio_dir):
    cmd = "python -m denoiser.enhance --dns48 --noisy_dir=" + str(audio_dir) +  " --out_dir=" + str(audio_dir) +  " --streaming -v"
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    enhance_audio('/Users/anirudhgupta/ekstepspeechrecognition/vakyansh-hf/market_mantra')

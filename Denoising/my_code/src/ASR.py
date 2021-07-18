from denoising import DenoisingModel
from common import Config
from tools import add_noise_to_audio
from utils import ensure_dir

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from pathlib import Path
import librosa
import jiwer
import argparse
import pandas as pd
import numpy as np
import random
from os.path import join
from tqdm import tqdm
import soundfile

ASR_MODEL_NAME = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

class ASR:
    def __init__(self, model_name):
        d = ModelDownloader()
        self.model = Speech2Text(# **d.download_and_unpack("asr_train_asr_conformer_raw_ru_bpe100_valid-2.zip"),
                                 **d.download_and_unpack(model_name),
                                 # Decoding parameters are not included in the model file
                                 maxlenratio=0.0,
                                 minlenratio=0.0,
                                 beam_size=20,
                                 ctc_weight=0.3,
                                 lm_weight=0.5,
                                 penalty=0.0,
                                 nbest=1)
    
    def recognate_speech(self, audio):
        nbests = self.model(audio)
        text, *_ = nbests[0]
        return text

    def compute_wer(self, gt_text, hypothesis):
        return jiwer.wer(gt_text, hypothesis)

def devide_audio(audio, duration, sr):
    cur_duration = librosa.get_duration(audio, sr=sr)
    if cur_duration < duration:
        result = np.zeros(duration*sr)
        result[:audio.shape[0]] = audio.copy()
        return [result]
    else:
        return [audio[:duration*sr]] + devide_audio(audio[duration*sr:], duration, sr)

def text_preprocessing(text):
    symbols = [',','.','?','!',';',':']
    text = text.lower()
    for symbol in symbols:
        text = text.replace(symbol, "")
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', dest='csv', help="csv file with audio paths and texts")
    parser.add_argument('--denoising', default=False, action='store_true', help="flag of using denoising algorithm")
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('--outputs', type=str, action='store')
    args = parser.parse_args()

    config = Config()

    outputs_root = join(config.project_root, args.outputs)
    ensure_dir(outputs_root)

    asr_model = ASR(ASR_MODEL_NAME)
    if args.denoising:
        denoising_model = DenoisingModel(config)
        denoising_model.load_weights(args.ckpt)

    data = pd.read_csv(join(config.data_root, args.csv))
    
    wer = {'noisy': [],
           'denoised': [],
           'original': []}

    for i, item in tqdm(data.iterrows()):
        save_dir = join(outputs_root, str(i))
        ensure_dir(save_dir)

        text = item['text']
        text = text_preprocessing(text)

        original_audio, _ = soundfile.read(item['clean_audio'], samplerate=config.sr)
        noise_src = random.choice([f.resolve() for f in Path(config.noise_src_test).rglob('*.wav')])
        noise = soundfile.read(noise_src, samplerate=config.sr)
        snr = np.random.choice(config.snrs, 1)[0]
        noisy_audio, _, _ = add_noise_to_audio(original_audio, noise, snr)
        # noisy_audio, _ = librosa.load(item['noisy_audio'], sr=config.sr)

        soundfile.write(join(save_dir, 'noisy.wav'), noisy_audio, config.sr)
        soundfile.write(join(save_dir, 'original.wav'), original_audio, config.sr)

        predicted_text = asr_model.recognate_speech(original_audio)
        predicted_text = text_preprocessing(predicted_text)
        wer['original'].append(asr_model.compute_wer(text, predicted_text))

        predicted_text = asr_model.recognate_speech(noisy_audio)
        predicted_text = text_preprocessing(predicted_text)
        wer['noisy'].append(asr_model.compute_wer(text, predicted_text))

        if args.denoising:
            audios = devide_audio(noisy_audio, config.duration, config.sr)
            cleaned_audio = np.empty(len(audios)*config.n_samples)

            for i, audio_part in enumerate(audios):
                denoised_part = denoising_model.predict(audio_part, data_type='raw')
                cleaned_audio[i*config.n_samples:(i+1)*config.n_samples] = \
                np.pad(denoised_part, (0,config.n_samples-denoised_part.shape[0]), 'mean')

            soundfile.write(join(save_dir, 'cleaned.wav'), cleaned_audio, config.sr)

            predicted_text = asr_model.recognate_speech(cleaned_audio)
            predicted_text = text_preprocessing(predicted_text)
            wer['denoised'].append(asr_model.compute_wer(text, predicted_text))

        print()
        print("SNR: ", snr)
        print("Noisy audio:    ", predicted_text)
        print("Denoised audio: ", predicted_text)
        print("Original audio: ", predicted_text)
        print("GT text:        ", text)
        print()
        print("Middle results:")
        for key in wer.keys():
            print("{}: {}".format(key, np.mean(wer[key])))
        print()

    print("Results:")
    for key in wer.keys():
        print("{}: {}".format(key, np.mean(wer[key])))


#TODO:
#add wers computing
#fix denoising prediction
#add denoising api

# Confirm the sampling rate is equal to that of the training corpus.
# If not, you need to resample the audio data before inputting to speech2text

# speech, rate = librosa.load()
# nbests = speech2text(speech)

# from os import listdir
# from os.path import join
# import numpy as np

# files = listdir(join(PROJECT_ROOT, 'results'))
# texts = listdir(join(PROJECT_ROOT, "texts"))

# mixes_wers = []
# clean_wers = []

# for text in texts:
#     filename = text.split('.')[0]
#     if filename + "-mixed.wav" in files:
#         mixed, _ = librosa.load(filename + "-mixed.wav")
#         clean, _ = librosa.load(filename + "-clean.wav")
#         with open(filename+".txt", "r") as f:
#             gt_text = f.readline()
#         mixed_text = speech2text(mixed)[0]
#         clean_text = speech2text(clean)[0]
        
#         mixed_wers.append(jiwer.wer(gt_text, mixed_text))
#         clean_wers.append(jiwer.wer(gt_text, clean_text))


# print('mixed_wer', np.mean(mixes_wers))
# print('clean_wer', np.mean(clean_wers))


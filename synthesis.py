import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import numpy as np
import torch
from hparams import create_hparams
from train import load_model
from text import text_to_sequence
import torchaudio
from scipy.io import wavfile
import os


def plot_data(path, data, figsize=(20, 6)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
    plt.savefig(path, format='png')
    plt.close()

###Setup hparams
hparams = create_hparams()

###Load model from checkpoint
checkpoint_path = "logs-LJSpeech-1.1/tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

mel2linear = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=80)
griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, n_iter=60, win_length=800, hop_length=200)

def txt2wav(num, text, output_path):
    mels_dir = os.path.join(output_path, 'mels')
    align_dir = os.path.join(output_path, 'align')
    wavs_dir = os.path.join(output_path, 'wavs')

    os.makedirs(mels_dir, exist_ok=True)
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(wavs_dir, exist_ok=True)

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    ###Decode text input and plot results
    _, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data(os.path.join(align_dir, num + '.png'),
              (mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))

    ###transform mel-spectrogram to waveform with griffin-lim using torchaudio packages
    mels = mel_outputs_postnet.float().data.cpu()[0]
    mels = np.power(10.0, mels)
    linear_spectrogram = mel2linear(mels)
    waveform = griffin_lim(linear_spectrogram)
    wavfile.write(os.path.join(wavs_dir, num + '.wav'), 16000, waveform.numpy())
    np.save(os.path.join(mels_dir, num + '.npy'), mel_outputs_postnet.float().data.cpu().numpy()[0], allow_pickle=False)

fread = open('/data07/xuehao/DATA/l2arctic/Model_data/adaptation_test_character_100')
output_path = 'tacotron_output_adaptation'
for line in fread.readlines():
    num, inference_data = line.strip().split('|', 1)
    print(os.path.basename(num))
    txt2wav((os.path.basename(num))[4:-4], inference_data, output_path)

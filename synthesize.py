from os.path import exists, join, expanduser

wn_preset = "20180510_mixture_lj_checkpoint_step000320000_ema.json"
#wn_checkpoint_path = "20180510_mixture_lj_checkpoint_step000320000_ema.pth"
wn_checkpoint_path = "checkpoints/checkpoint_step000320000_ema.pth"
#wn_checkpoint_path = "checkpoints/checkpoint_step000325000_ema.pth"

import librosa.display
import IPython
from IPython.display import Audio
import numpy as np
import torch

# Setup WaveNet vocoder hparams
from hparams import hparams
with open(wn_preset) as f:
    hparams.parse_json(f.read())

# Setup WaveNet vocoder
from train import build_model
from synthesis import wavegen
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = build_model().to(device)

print("Load checkpoint from {}".format(wn_checkpoint_path))
checkpoint = torch.load(wn_checkpoint_path)
model.load_state_dict(checkpoint["state_dict"])

from glob import glob
from tqdm import tqdm

with open("../Tacotron-2/tacotron_output/eval/map.txt") as f:
    maps = f.readlines()
	
maps = list(map(lambda x:x[:-1].split("|"), maps))
# filter out invalid ones
maps = list(filter(lambda x:len(x) == 2, maps))
	
print("List of texts to be synthesized")
for idx, (text,_) in enumerate(maps):
    print(idx, text)
	
waveforms = []

for idx, (text, mel) in enumerate(maps):
    print("\n", idx, text)
    mel_path = join("../Tacotron-2", mel)
    c = np.load(mel_path)
    if c.shape[1] != hparams.num_mels:
        np.swapaxes(c, 0, 1)
    # Range [0, 4] was used for training Tacotron2 but WaveNet vocoder assumes [0, 1]
    c = np.interp(c, (0, 4), (0, 1))
 
    # Generate
    waveform = wavegen(model, c=c, fast=True, tqdm=tqdm)
  
    waveforms.append(waveform)
	

librosa.output.write_wav("out.wav", waveforms[0], hparams.sample_rate)

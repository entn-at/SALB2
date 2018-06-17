#/bin/bash

PYTHON=python3
base_dir=$PWD
wavenet_dir="wavenet_vocoder"
taco2_dir="Tacotron-2"

wavenet_config=$PWD/data/wavenet_config.json

# get wavenet and tacotron repo
rm -rf $wavenet_dir
rm -rf $taco2_dir
git clone https://github.com/m-toman/$wavenet_dir
git clone https://github.com/m-toman/$taco2_dir

# copy conf
cp data/tacotron_hparams.py $taco2_dir/hparams.py

echo "To run Tacotron-2 training:"
echo "cd $taco2_dir"
echo "$PYTHON train.py"

# synthesize a test stencne
#cd $taco2_dir
#echo "This is really awesome, let's do it!" > text_list.txt
#cat text_list.txt

#rm -rf tacotron_output
#python3 synthesize.py --model='Tacotron' --mode='eval' \
#      --hparams='symmetric_mels=False,max_abs_value=4.0,power=1.1,outputs_per_step=1' \
#      --text_list=./text_list.txt


# adapt to get wavs
# cd $taco2_dir
# mkdir LJSpeech-1.1
# for i in /mnt/raw/10498/wav/*.wav; do sox -V1 --norm $i -r 22050 -b 16 wavs/`basename $i` channels 1; done
#  cat input.txt | sed 's/"//' | sed 's/","/||/' | sed  's/[^a-zA-Z.?!]\+"$//' > metadata.csv


# train taco
#cd $taco2_dir
#python3 preprocess.py
#python3 train.py --checkpoint_interval=250
# cd ..


# preprocess wavenet
#cd $wavenet_dir
#cp -r ../$taco2_dir/LJSpeech-1.1 .
# python3 preprocess.py --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json ljspeech /mnt/working/markus/tacowavenet/wavenet_vocoder/LJSpeech-1.1/ $PWD/training_data

# train wavenet
#python3 train.py --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json --data-root=./training_data/ --checkpoint=20180510_mixture_lj_checkpoint_step000320000_ema.pth

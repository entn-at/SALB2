mkdir tacowavenet
cd tacowavenet

# get wavenet repo
wavenet_dir="wavenet_vocoder"
git clone https://github.com/r9y9/$wavenet_dir
    
# get tacotron repo
taco2_dir="Tacotron-2"
git clone https://github.com/r9y9/$taco2_dir
cd $taco2_dir
git checkout -B wavenet3 origin/wavenet3

# get pretrained tacotron model
mkdir -p logs-Tacotron
curl -O -L "https://www.dropbox.com/s/vx7y4qqs732sqgg/pretrained.tar.gz"
tar xzvf pretrained.tar.gz
mv pretrained logs-Tacotron

cd ..

# get pretrained wavnet
cd $wavenet_dir
wn_preset="20180510_mixture_lj_checkpoint_step000320000_ema.json"
wn_checkpoint_path="20180510_mixture_lj_checkpoint_step000320000_ema.pth"
curl -O -L "https://www.dropbox.com/s/0vsd7973w20eskz/20180510_mixture_lj_checkpoint_step000320000_ema.json"
curl -O -L "https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth"

cd ..

# synthesize a test stencne
cd $taco2_dir
#echo "Change will not come if we wait for some other person or some other time, We are the ones we've been waiting for, We are the change that we seek" > text_list.txt
echo "This is really awesome, let's do it!" > text_list.txt
cat text_list.txt

rm -rf tacotron_output
python3 synthesize.py --model='Tacotron' --mode='eval' \
      --hparams='symmetric_mels=False,max_abs_value=4.0,power=1.1,outputs_per_step=1' \
      --text_list=./text_list.txt


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

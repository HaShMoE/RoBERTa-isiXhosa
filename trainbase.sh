#!/bin/bash

#PBS -N train_model
#PBS -q gpu_1
#PBS -l select=1:ncpus=5:ngpus=1
#PBS -l walltime=12:00:00
#PBS -P CSCI1674
#PBS -m abe
#PBS -WMail_Users=mhmhis005@uct.ac.za
ulimit -s unlimited

module load chpc/python/anaconda/3-2021.11
#module load chpc/cuda/11.6/SXM2/11.6
module load chpc/openmpi/4.1.1/gcc-6.1.0
source /home/hmahomed/myenv/bin/activate

wandb login
WANDB_PROJECT=roberta_trained_model
WANDB_LOG_MODEL="checkpoint"

cd /home/hmahomed/lustre/roberta || exit

now=$(date +%Y%m%d-%H:%M:%S:%3N)

start=`date +%s`

# Run Python Script
echo "Starting" 
#pip3 install --user -r requirements.txt
python3 train_tokenizer.py
echo "Training"
python3 run_mlm.py \
        --model_type roberta \
        --tokenizer_name ./ \
        --config_name ./ \
        --train_file ./wura-xh/train.txt \
        --validation_file ./wura-xh/valid.txt \
        --do_train \
        --do_eval \
        --num_train_epochs 20 \
        --save_steps 1000 \
        --output_dir ./output20

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with unspecified cores was $runtime minutes."



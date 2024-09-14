#!/bin/bash

#PBS -N train_nr
#PBS -q gpu_1
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -l walltime=11:00:00
#PBS -P CSCI1674
#PBS -m abe
#PBS -WMail_Users=mhmhis005@uct.ac.za
ulimit -s unlimited

module load chpc/python/anaconda/3-2021.11
module load chpc/cuda/11.6/SXM2/11.6
source /home/hmahomed/myenv/bin/activate

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
        --max_seq_length 256 \
        --do_train \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--num_train_epochs 1 \
	--output_dir ./output

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with unspecified cores was $runtime minutes."

#!/bin/bash
ulimit -s unlimited

now=$(date +%Y%m%d-%H:%M:%S:%3N)

start=`date +%s`

# Run Python Script
echo "Starting" 
pip3 install -r requirements.txt
python3 train_tokenizer.py
echo "Training"
python3 RoBERTa-isiXhosa/code/run_mlm.py \
        --model_type roberta \
        --tokenizer_name ./model \
        --config_name ./model \
        --train_file ./wura-xh/train.txt \
        --validation_file ./wura-xh/valid.txt \
        --do_train \
        --do_eval \
        --num_train_epochs 20 \
        --save_steps 1000 \
        --output_dir ./output

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with unspecified cores was $runtime minutes."



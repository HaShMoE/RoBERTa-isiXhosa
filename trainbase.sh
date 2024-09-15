#!/bin/bash
ulimit -s unlimited

now=$(date +%Y%m%d-%H:%M:%S:%3N)

start=`date +%s`

export TOKENIZER=/RoBERTa-isiXhosa/model
export CONFIG=/RoBERTa-isiXhosa/model
export OUTPUT_DIR=/RoBERTa-isiXhosa/output/training
export TRAIN=RoBERTa-isiXhosa/dataset/train.txt
export VALID=RoBERTa-isiXhosa/dataset/valid.txt
export NUM_EPOCHS=20
export SAVE_STEPS=1000

# Run Python Script
echo "Starting" 
pip3 install -r requirements.txt
python3 train_tokenizer.py
echo "Training"
python3 RoBERTa-isiXhosa/training/run_mlm.py \
        --model_type roberta \
        --tokenizer_name $TOKENIZER \
        --config_name $CONFIG \
        --train_file $TRAIN \
        --validation_file $VALID \
        --do_train \
        --do_eval \
        --num_train_epochs $NUM_EPOCHS \
        --save_steps $SAVE_STEPS \
        --output_dir $OUTPUT_DIR

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with unspecified cores was $runtime minutes."



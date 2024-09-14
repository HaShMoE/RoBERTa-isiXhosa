#!/bin/bash

#PBS -N finetune_model
#PBS -q gpu_1
#PBS -l select=1:ncpus=7:ngpus=1
#PBS -l walltime=8:00:00
#PBS -P CSCI1674
#PBS -m abe
#PBS -WMail_Users=mhmhis005@uct.ac.za
ulimit -s unlimited

module load chpc/python/anaconda/3-2021.11
module load chpc/cuda/11.6/SXM2/11.6
module load chpc/openmpi/4.1.1/gcc-6.1.0
source /home/hmahomed/myenv/bin/activate

cd /home/hmahomed/lustre/roberta || exit

now=$(date +%Y%m%d-%H:%M:%S:%3N)

start=`date +%s`

LANG=xho
for j in 1 2 3 4 5
do
        export MAX_LENGTH=256
        export ROBERTA_MODEL=/home/hmahomed/lustre/roberta/output/
        export OUTPUT_DIR=/home/hmahomed/lustre/roberta/finetuned/NEWS/
        export BATCH_SIZE=16
        export NUM_EPOCHS=20
        export SAVE_STEPS=10000
        export SEED=$j
        export TEST_RESULT=test_result$j.txt
        export TEST_PRED=test_pred$j.txt

        # Run Python Script
        echo "Starting$j"
        python3 /home/hmahomed/lustre/masakhane-news/code/train_textclass.py \
                --data_dir /home/hmahomed/lustre/masakhane-news/data/${LANG}/ \
                --model_type roberta \
                --model_name_or_path $ROBERTA_MODEL \
                --output_result $TEST_RESULT \
                --output_prediction_file $TEST_PRED \
                --do_train \
                --do_predict \
                --learning_rate 2e-5 \
                --per_gpu_train_batch_size $BATCH_SIZE \
                --per_gpu_eval_batch_size $BATCH_SIZE \
                --max_seq_length  $MAX_LENGTH \
                --gradient_accumulation_steps 2 \
                --num_train_epochs $NUM_EPOCHS \
                --save_steps $SAVE_STEPS \
                --output_dir $OUTPUT_DIR \
                --seed $SEED \
                --overwrite_output_dir &>> "/home/hmahomed/lustre/roberta/finetuned/log_finetune_NEWS"
done

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with unspecified cores was $runtime minutes."


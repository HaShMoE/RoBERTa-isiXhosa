#!/bin/bash

now=$(date +%Y%m%d-%H:%M:%S:%3N)

start=`date +%s`

LANG=xho
pip3 install -r requirements.txt

for j in 1 2 3 4 5
do
        export MAX_LENGTH=256
        export ROBERTA_MODEL=/RoBERTa-isiXhosa/output/
        export OUTPUT_DIR=/RoBERTa-isiXhosa/finetuned/NEWS/
        export BATCH_SIZE=16
        export NUM_EPOCHS=20
        export SAVE_STEPS=10000
        export SEED=$j
        export TEST_RESULT=test_result$j.txt
        export TEST_PRED=test_pred$j.txt

        # Run Python Script
        echo "Starting$j"
        python3 /RoBERTa-isiXhosa/masakhane-news/code/train_textclass.py \
                --data_dir /RoBERTa-isiXhosa/masakhane-news/data/${LANG}/ \
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
                --overwrite_output_dir &>> "/RoBERTa-isiXhosa/finetuned/log_finetune_NEWS"
done

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with unspecified cores was $runtime minutes."


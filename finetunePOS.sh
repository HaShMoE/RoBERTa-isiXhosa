#!/bin/bash
ulimit -s unlimited

module load chpc/python/anaconda/3-2021.11
#module load chpc/cuda/11.6/SXM2/11.6
module load chpc/openmpi/4.1.1/gcc-6.1.0
source /home/hmahomed/myenv/bin/activate

now=$(date +%Y%m%d-%H:%M:%S:%3N)

start=`date +%s`

LANG=xho

pip3 install -r requirements.txt

for j in 1 2 3 4 5
do
        export MAX_LENGTH=200
        export ROBERTA_MODEL=RoBERTa-isiXhosa/output
        export OUTPUT_DIR=RoBERTa-isiXhosa/finetuned/POS/
        export BATCH_SIZE=16
        export NUM_EPOCHS=20
        export SAVE_STEPS=10000
        export SEED=$j
        export TEST_RESULT=test_result$j.txt
        export TEST_PRED=test_pred$j.txt

        # Run Python Script
        echo "Starting"
        python3 /RoBERTa-isiXhosa/masakhane-pos/train_pos.py \
                --data_dir /RoBERTa-isiXhosa/masakhane-pos/data/${LANG}/ \
                --model_type roberta \
                --model_name_or_path $ROBERTA_MODEL \
                --test_result_file $TEST_RESULT \
		--test_prediction_file $TEST_PRED \
                --max_seq_length  $MAX_LENGTH \
                --per_gpu_train_batch_size $BATCH_SIZE \
                --gradient_accumulation_steps 2 \
                --do_train \
                --do_eval \
                --do_predict \
                --num_train_epochs $NUM_EPOCHS \
                --save_steps $SAVE_STEPS \
                --output_dir $OUTPUT_DIR \
                --seed $SEED \
                --overwrite_output_dir &>> "/RoBERTa-isiXhosa/finetuned/log_finetune_POS"
done

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with unspecified cores was $runtime minutes."



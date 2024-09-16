#!/bin/bash

ulimit -s unlimited

now=$(date +%Y%m%d-%H:%M:%S:%3N)

start=`date +%s`

LANG=xho
for j in 1 2 3 4 5
do
	export MAX_LENGTH=164
	export ROBERTA_MODEL=/RoBERTa-isiXhosa/output/training
	export OUTPUT_DIR=/RoBERTa-isiXhosa/output/finetuning/NER
	export BATCH_SIZE=32
	export NUM_EPOCHS=20
	export SAVE_STEPS=1000
	export SEED=$j
	export TEST_RESULT=test_result$j.txt
	export TEST_PRED=test_pred$j.txt

	# Run Python Script
	echo "Starting"
	python3 /RoBERTa-isiXhosa/code/tokenization/train_ner.py \
		--data_dir /RoBERTa-isiXhosa/masakhane-ner/MasakhaNER2.0/data/${LANG}/ \
        	--model_type roberta \
        	--model_name_or_path $ROBERTA_MODEL \
		--test_result_file $TEST_RESULT \
		--test_prediction_file $TEST_PRED \
        	--do_train \
        	--do_eval \
        	--do_predict \
		--num_train_epochs $NUM_EPOCHS \
        	--save_steps $SAVE_STEPS \
        	--output_dir $OUTPUT_DIR \
		--seed $SEED \
		--overwrite_output_dir &>> "/RoBERTa-isiXhosa/output/finetuning/NER/log_finetune_NER"
done

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with unspecified cores was $runtime minutes."



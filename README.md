# Train RoBERTa from scratch 

## Prerequisites
All required packages are listed in the requirements.txt file and are installed in each of the bash scripts.

## Data
The data to be used for training 

## Tokenization
A BPE tokenizer is used from the Huggingface tokenizers library. The code/tokenization/train_tokenizer.py script was created to train a BPE tokenizer on the training data. The trainbase.sh script trains a tokenizer and saves the configuration in the model directory.

## Training
The run_mlm.py script from the Huggingface transformers library is used to train the RoBERTa model from scratch. This script is called from the trainbase.sh bash script which also specifies all the hyperparameters used in training.

## Finetuning 
Three bash scripts, i.e. finetuneNER, finetuneNEWS and finetunePOS, are used for finetuning. The python scripts called were taken from the Masakhane datasets for each task.

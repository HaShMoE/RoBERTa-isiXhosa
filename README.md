# Train RoBERTa from scratch 

## Prerequisites
All required packages are listed in the requirements.txt file and are installed in each of the bash scripts. This can be doen using the following command.
```shell
pip -r install requirements.txt
```
The other requirements are 
* python 3
* cuda 11.6

## Data
### Training
The data used for training was taken from the isiXhosa portion of the wura dataset and seperated into a training, validation and test set. This is in the data directory.

### Finetuning
The Masakhane datasets for fintuning are to be cloned in the RoBERTa-isiXhosa directory as shown in the code below.

```shell
cd RoBERTa-isiXhosa
git clone https://github.com/masakhane-io/masakhane-ner.git
git clone https://github.com/masakhane-io/masakhane-news.git
git clone https://github.com/masakhane-io/masakhane-pos.git
```

The language used for finetuning can be specified in the finetuning bash scripts by chnaging the LANG variable from 'xho' (isiXhosa) to the desired language.

## Tokenization
A BPE tokenizer is used from the Huggingface tokenizers library. The code/tokenization/train_tokenizer.py script was created to train a BPE tokenizer on the training data. The trainbase.sh script trains a tokenizer and saves the configuration in the model directory.

## Training
The run_mlm.py script from the Huggingface transformers library is used to train the RoBERTa model from scratch. This script is called from the trainbase.sh bash script which also specifies all the hyperparameters used in training.

## Finetuning 
Three bash scripts, i.e. finetuneNER, finetuneNEWS and finetunePOS, are used for finetuning. The python scripts called were taken from the Masakhane datasets for each task. They have been altered to produce more in-depth results for analysis. These altered scripts can be fond in the code/finetuning directory.

# README for Homework 1 ADL NTU 2022 Spring

## Environment
intent_train.py/slot_train.py use livelossplot package to plot
install livelossplot package : pip install livelossplot

## Preprocessing
Use command bash preprocess.sh to get glove.840B.300d.txt
Then it will run python intent_preprocess.py and python slot_preprocess.py to pre-process for the two tasks respectively

intent_preprocess.py --train-dataset-path <train-dataset-path> --valid-dataset-path <valid-dataset-path> --test-dataset-path <test-dataset-path> --preprocess-dir <preprocess-dir> --max-words <max-words> --max-len <max-len>


slot_preprocess.py --train-dataset-path <train-dataset-path> --valid-dataset-path <valid-dataset-path> --test-dataset-path <test-dataset-path> --preprocess-dir <preprocess-dir> --max-words <max-words> --max-len <max-len>

## Intent detection

train: 

python intent_train.py --preprocess-dir <preprocess-dir> --ckpt-save-path <ckpt-save-path> --max-words <max-words> --max-len <max-len> --model <model> --lr <lr> --batch-size <batch-size> --epoch <epoch>

test: 

python intent_test.py --test-dataset-path <test-dataset-path> --preprocess-dir <preprocess-dir> --pred-file-path <pred-file-path> --max-words <max-words> --max-len <max-len> --model <model> --ckpt-load-path <ckpt-load-path>

Use command bash ./intent_cls.sh ./Path/to/the/test/file ./Path/to/the/predict/file to reproduce best result

## slot tagging

train: 

python intent_train.py --preprocess-dir <preprocess-dir> --ckpt-save-path <ckpt-save-path> --max-words <max-words> --max-len <max-len> --model <model> --lr <lr> --batch-size <batch-size> --epoch <epoch>

test: 

python intent_test.py --test-dataset-path <test-dataset-path> --valid-dataset-path <valid-dataset-path>--preprocess-dir <preprocess-dir> --pred-file-path <pred-file-path> --max-words <max-words> --max-len <max-len> --model <model> --ckpt-load-path <ckpt-load-path>

Use command bash ./slot_tag.sh ./Path/to/the/test/file ./Path/to/the/predict/file to reproduce best result


##argument instructions

train-dataset-path : Path to the train dataset.
valid-dataset-path : Path to the validation dataset.
test-dataset-path : Path to the test dataset.
ckpt-save-path : Path to the save checkpoint file. 
ckpt-load-path : Path to the load checkpoint file.
pred-file-path: Path to the create predict file.
preprocess-dir : Directory to the preprocessed files.
max-words : Vocabulary size of train/validation/test dataset.
max-len : Maximum length of a sequence.
model : Network model to train/validation/test.
lr : Learning rate.
batch-size : Batch size.
epoch : Maximum training epoch.



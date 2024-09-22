# README for Homework 2 ADL NTU 2022 Spring

## Environment

plot_loss_em.py use matplotlib package to plot
install matplotlib package : pip install matplotlib

## Preprocessing

Multiple Choice task:

convert train/valid data to SWAG format:
python3.9 data2swag_train_valid.py --train_file <--train_file> --validation_file <validation_file> --context_file <context_file>

--train_file : The input training data file (a text file).
--validation_file : An optional input evaluation data file to evaluate the perplexity on (a text file).
--context_file : An optional input context data file to evaluate the perplexity on (a text file).


convert test data to SWAG format:
python3.9 data2swag_test.py --test_file <test_file> --context_file <context_file>

--test_file : An optional input testing data file to test the perplexity on (a text file).
--context_file : An optional input context data file to evaluate the perplexity on (a text file).



Question Answering task:

convert train/valid data to SQuAD format:
python3.9 data2SQuAD_train_valid.py

--train_file : The input training data file (a text file).
--validation_file : An optional input evaluation data file to evaluate the perplexity on (a text file).
--context_file : An optional input context data file to evaluate the perplexity on (a text file).

Use the result of multiple choice task as input ,and convert it to SQuAD data format:
python3.9 data2SQuAD_test.py --context_file <context_file> --MC_output_path <MC_output_path> --model_name <model_name>

--context_file : An optional input context data file to evaluate the perplexity on (a text file).
--MC_output_path : Path to MC output file
--model_name : Hugging face model name ,remove '/',and replace '-' with '_'


## Multiple Choice task

train: 


python3.9 run_MC.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --cache_dir <cache_dir> \
  --train_from_scratch <train_from_scratch> \
  --do_train \
  --do_eval \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --pad_to_max_length \
  --max_seq_length <max_seq_length> \
  --learning_rate <learning_rate> \
  --num_train_epochs <num_train_epochs> \
  --per_device_train_batch_size <per_device_train_batch_size> \
  --gradient_accumulation_steps <gradient_accumulation_steps>\
  --per_device_eval_batch_size <per_device_eval_batch_size> \
  --eval_accumulation_steps <eval_accumulation_steps> \
  --overwrite_output \
  --logging_steps <logging_steps> \
  --save_strategy <save_strategy> \
  --evaluation_strategy <evaluation_strategy> \
  --eval_steps <eval_steps> \
  --warmup_ratio <warmup_ratio>
  
test: 

python3.9 run_MC.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --output_file <output_file> \
  --cache_dir <cache_dir> \
  --pad_to_max_length \
  --test_file <test_file> \
  --do_predict \
  --max_seq_length <max_seq_length> \
  --per_device_eval_batch_size <per_device_eval_batch_size>
  
Use commands python3.9 run_MC.py -h for the list of optional arguments 
  


## Question Answering task

train: 

python3.9 run_QA.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --cache_dir <cache_dir> \
  --do_train \
  --do_eval \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --per_device_train_batch_size <per_device_train_batch_size> \
  --gradient_accumulation_steps <gradient_accumulation_steps>\
  --per_device_eval_batch_size <per_device_eval_batch_size> \
  --eval_accumulation_steps <eval_accumulation_steps> \
  --learning_rate <learning_rate> \
  --num_train_epochs <num_train_epochs> \
  --max_seq_length <max_seq_length> \
  --doc_stride <doc_stride> \
  --warmup_ratio <warmup_ratio> \
  --overwrite_output \
  --logging_steps <logging_steps> \
  --save_strategy <save_strategy> \
  --evaluation_strategy <evaluation_strategy> \
  --eval_steps <eval_steps> \

test: 

python3.9 run_QA.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --output_file <output_file> \
  --cache_dir <cache_dir> \
  --test_file <test_file> \
  --pad_to_max_length \
  --do_predict \
  --max_seq_length <max_seq_length> \
  --doc_stride <doc_stride> \
  --per_device_eval_batch_size <per_device_eval_batch_size> \

Use commands python3.9 run_QA.py -h for the list of optional arguments 

## Reproduce
Use command bash ./run.sh ./path/to/context.json /path/to/test.json  ./path/to/pred/prediction.csv to reproduce best result

## Plot learning curve of EM and evaluation loss(QA task)
python3.9 plot_loss_em.py --trainer_state_path <trainer_state_path>

--trainer_state_path : Path to the trainer_state json file.








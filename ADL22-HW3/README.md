# README for Homework 3 ADL NTU 2022 Spring

## Environment

plot_learning_curve.py use matplotlib package to plot
install matplotlib package : pip install matplotlib


## Train/validate

### 用哪一種generation strategy 不影響training 結果(model 參數更新),但會影響Rouge score

python3.9 run_summarization.py \
  --do_train \
  --do_eval \
  --model_name_or_path google/mt5-small \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --output_dir <output_dir>  \
  --logging_nan_inf_filter=True \
  --logging_strategy='epoch' \
  --load_best_model_at_end=True \
  --evaluation_strategy='epoch' \
  --save_strategy='epoch' \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --per_device_eval_batch_size=4 \
  --eval_accumulation_steps=8 \
  --predict_with_generate=True \
  --text_column maintext \
  --summary_column title \
  --num_train_epochs=5 \
  --save_total_limit=3 \
  --adafactor=True \
  --learning_rate 1e-3 \
  --warmup_ratio 0.1 \
  [--num_beams <num_beams>] \
  [--do_sample] \
  [--top_k <top_k>] \
  [--top_p <top_p>] \
  [--temperature <temperature>] \
  [--repetition_penalty <repetition_penalty>] \

  
train_file: path to training data file. e.g., ./data/train.jsonl
validation_file: path to validation data file. e.g., ./data/public.jsonl
output_dir: The output directory where the model predictions and checkpoints will be written. e.g., ./BeamSearch_numbeams5
num_beams: Number of beams to use for decoding. e.g., 5
do_sample: Whether or not to use sampling ; use greedy decoding otherwise.
top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering. e.g., 25
top_p: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. e.g., 0.9
temperature: The value used to module the next token probabilities. e.g., 0.75
repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty. e.g., 2.0


Use commands python3.9 run_summarization.py -h for the list of optional arguments 

## Test

python3.9 run_summarization.py \
  --do_predict \
  --model_name_or_path <model_name_or_path> \
  --test_file <test_file> \
  --output_file <output_file> \
  --output_dir <output_dir> \
  --predict_with_generate \
  --text_column maintext \
  --summary_column title \
  --per_device_eval_batch_size 4 \
  [--num_beams <num_beams>] \
  [--do_sample] \
  [--top_k <top_k>] \
  [--top_p <top_p>] \
  [--temperature <temperature>] \
  [--repetition_penalty <repetition_penalty>] \


model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models. e.g., ./best_ckpt
test_file: path to testing data file. e.g., ./data/public.jsonl
output_file: Path to prediction file. e.g., ./pred.jsonl
output_dir: The output directory where the model predictions and checkpoints will be written. e.g., ./best_ckpt
num_beams: Number of beams to use for decoding. e.g., 5
do_sample: Whether or not to use sampling ; use greedy decoding otherwise.
top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering. e.g., 25
top_p: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. e.g., 0.9
temperature: The value used to module the next token probabilities. e.g., 0.75
repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty. e.g., 2.0


Use commands python3.9 run_summarization.py -h for the list of optional arguments



## Evaluation

Use the Script:

usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION


Example:

python3.9 eval.py -r public.jsonl -s submission.jsonl
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
 

## Reproduce
Use following commands:
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl

to reproduce best result

## Plot learning curve of ROUGE
python3.9 plot_learning_curve.py --trainer_state_path <trainer_state_path>

--trainer_state_path : Path to the trainer_state json file.








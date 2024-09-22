#Hugging face model name ,remove '/',and replace '-' with '_'
model=chinese_roberta_wwm_ext

#convert test data to SWAG form
python3.9 data2swag_test.py --test_file "${2}" --context_file "${1}"

#MC prediction
python3.9 run_MC.py \
  --model_name_or_path ./$model/mc/ \
  --output_dir ./$model/mc/ \
  --output_file ./$model/mc/prediction.json \
  --cache_dir ./$model/cache/ \
  --pad_to_max_length \
  --test_file ./preprocess/mc/test_swag_format.csv \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 1
  
#convert test data to SQuAD form
python3.9 data2SQuAD_test.py --context_file "${1}" --MC_output_path ./$model/mc/prediction.json --model_name $model
  
#QA prediction  
python3.9 run_QA.py \
  --model_name_or_path ./$model/qa/ \
  --output_dir ./$model/qa/ \
  --output_file "${3}" \
  --cache_dir ./$model/cache/ \
  --test_file ./preprocess/qa/test/"$model"_test_SQuAD_format.json \
  --pad_to_max_length \
  --do_predict \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_eval_batch_size 1  

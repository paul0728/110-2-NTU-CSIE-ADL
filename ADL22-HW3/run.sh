python3.9 run_summarization.py \
  --do_predict \
  --model_name_or_path ./best_ckpt \
  --test_file "${1}" \
  --output_file "${2}" \
  --output_dir ./best_ckpt \
  --predict_with_generate \
  --text_column maintext \
  --summary_column title \
  --per_device_eval_batch_size 4 \
  --num_beams 5 \

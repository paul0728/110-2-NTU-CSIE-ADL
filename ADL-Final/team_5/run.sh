pip install -r requirements.txt
gdown --id 176JsJk0cJHg1Gn6CDJwbYqYmRBFuQw3K -O bot/pytorch_model.bin
python simulator.py --model_name_or_path ./bot --split test --num_chats 980
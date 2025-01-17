export CUDA_VISIBLE_DEVICES=0

model_name=FCM

python -u /home/wwr/zsn/FCM/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path MSL \
  --model_id MSL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 100 \
  --label_len 50 \
  --pred_len 100 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --dataset MSL
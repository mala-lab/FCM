export CUDA_VISIBLE_DEVICES=3

for i in {1..250};
do
model_name=FCM

python -u /home/wwr/zsn/FCM/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path UCR \
  --model_id UCR_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 30 \
  --step_len 50 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --batch_size 20 \
  --learning_rate 0.001 \
  --itr 1 \
  --dataset UCR\
  --index $i

done  
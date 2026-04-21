export CUDA_VISIBLE_DEVICES=6,5

train_epochs=100
learning_rate=0.001 #0.001
llm_layers=2

master_port=00081
num_process=2
batch_size=8
d_model=16
d_ff=128 
dropout=0.05

description="The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. The time resolution is hourly."

for model_name in SLAN
do
for d_state in 64 
do
for pred_len in 96 
do

nohup accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_state $d_state \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment ETTh1$model_name  > $model_name'pred_len'$pred_len'_d_state'$d_state.log 2>&1

done
done
done
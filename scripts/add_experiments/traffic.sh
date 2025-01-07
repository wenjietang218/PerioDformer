if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

seq_len=336
model_name=PerioDformer

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2024

for C in 42
do
  python -u run_longExp.py \
      --percent 5 \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_96' \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len 96 \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --d_model2 64 \
      --d_ff2 256 \
      --w 1\
      --C $C\
      --mlp_num 1\
      --experiment 2\
      --dimension 24\
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 80\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 16 --learning_rate 0.0001
done
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

#24
seq_len=336
model_name=PerioDformer

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

random_seed=2024

for C in 12 48
do
  for pred_len in 96 192 336 720 960
  do
    python -u run_longExp.py \
        --percent 10 \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 321 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --d_model2 64 \
        --d_ff2 128 \
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
        --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
  done
done

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

seq_len=384
model_name=PerioDformer

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2024

#384_960
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_960' \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 960 \
    --enc_in 7 \
    --n_heads 8 \
    --e_layers 5 \
    --w 6 \
    --mlp_num 1 \
    --d_model 64 \
    --d_ff 128 \
    --d_model2 128 \
    --d_ff2 256 \
    --C 48\
    --experiment 2\
    --dimension 64 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --des 'Exp' \
    --train_epochs 60\
    --patience 20\
    --lradj 'TST'\
    --pct_start 0.4\
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_960'.log




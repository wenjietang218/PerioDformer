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
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2024

#64-128 128-256

#336_96 2-5-1
python -u run_longExp.py \
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
    --enc_in 7 \
    --n_heads 8 \
    --e_layers 2 \
    --w 8 \
    --mlp_num 3 \
    --d_model 32 \
    --d_ff 128 \
    --d_model2 256 \
    --d_ff2 256 \
    --C 192\
    --experiment 2\
    --dimension 104 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --des 'Exp' \
    --train_epochs 10\
    --patience 20\
    --lradj 'TST'\
    --pct_start 0.4\
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_96'.log

#336_192 2-3-1
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_192' \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 192 \
    --enc_in 7 \
    --n_heads 8 \
    --e_layers 2 \
    --w 1 \
    --mlp_num 4 \
    --d_model 32 \
    --d_ff 128 \
    --d_model2 256 \
    --d_ff2 256 \
    --C 192\
    --experiment 2\
    --dimension 104 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --des 'Exp' \
    --train_epochs 10\
    --patience 20\
    --lradj 'TST'\
    --pct_start 0.4\
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_192'.log

#336_336 2-6-1
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_336' \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 336 \
    --enc_in 7 \
    --n_heads 8 \
    --e_layers 3 \
    --w 3 \
    --mlp_num 2 \
    --d_model 32 \
    --d_ff 128 \
    --d_model2 256 \
    --d_ff2 256 \
    --C 192\
    --experiment 2\
    --dimension 104 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --des 'Exp' \
    --train_epochs 10\
    --patience 20\
    --lradj 'TST'\
    --pct_start 0.4\
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_336'.log

#336_720 5-6-1
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_720' \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 720 \
    --enc_in 7 \
    --n_heads 8 \
    --e_layers 5 \
    --w 1 \
    --mlp_num 5 \
    --d_model 32 \
    --d_ff 128 \
    --d_model2 256 \
    --d_ff2 256 \
    --C 192\
    --experiment 2\
    --dimension 104 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --des 'Exp' \
    --train_epochs 10\
    --patience 20\
    --lradj 'TST'\
    --pct_start 0.4\
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_720'.log

#336_960 5-6-1
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
    --w 1 \
    --mlp_num 4 \
    --d_model 32 \
    --d_ff 128 \
    --d_model2 256 \
    --d_ff2 256 \
    --C 192\
    --experiment 2\
    --dimension 104 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --des 'Exp' \
    --train_epochs 10\
    --patience 20\
    --lradj 'TST'\
    --pct_start 0.4\
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_960'.log




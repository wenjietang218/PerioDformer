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
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2024

for root_path_name in ./dataset/1%/  ./dataset/5%/  ./dataset/10%/
do
  #336_96
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
      --n_heads 4 \
      --e_layers 1 \
      --w 3 \
      --mlp_num 2 \
      --d_model 16 \
      --d_ff 16 \
      --d_model2 32 \
      --d_ff2 32 \
      --C 24\
      --experiment 2\
      --dimension 24 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 60\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_96'.log

  #336_192
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
      --n_heads 4 \
      --e_layers 1 \
      --w 3 \
      --mlp_num 1 \
      --d_model 16 \
      --d_ff 16 \
      --d_model2 32 \
      --d_ff2 32 \
      --C 24\
      --experiment 2\
      --dimension 24 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 60\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_192'.log

  #336_336
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
      --n_heads 4 \
      --e_layers 2 \
      --w 5 \
      --mlp_num 1 \
      --d_model 16 \
      --d_ff 16 \
      --d_model2 32 \
      --d_ff2 32 \
      --C 24\
      --experiment 2\
      --dimension 24 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 60\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_336'.log

  #336_720
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
      --n_heads 4 \
      --e_layers 3 \
      --w 4 \
      --mlp_num 1 \
      --d_model 16 \
      --d_ff 16 \
      --d_model2 32 \
      --d_ff2 32 \
      --C 24\
      --experiment 2\
      --dimension 24 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 60\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_720'.log

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
      --n_heads 4 \
      --e_layers 3 \
      --w 3 \
      --mlp_num 1 \
      --d_model 16 \
      --d_ff 16 \
      --d_model2 32 \
      --d_ff2 32 \
      --C 24\
      --experiment 2\
      --dimension 24 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 60\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_960'.log
done





# PerioDformer
We propose periodic Disposition Enhanced Transformer (PeriD
former) for time series forecasting, which transforming input series into input
 tokens with more semantic meaning.
Experimental data show that PeriDformer achieves state-of-the
art performance on six challenging real-world datasets.

# Model Structure
Specifically, PerioDformer contains three key designs:

- **Periodic Disposition**: PerioDformer first analyzes the natural periodicity of the sequence and subsequently partitions the time series into period blocks based on this identified natural period. We concatenate the periodic blocks into tokens for two vanilla Transformer encoders, which contain self-attention architecture.
- **Intra-period and Inter-period Encoders**: Intra-period tokens are the input for the intra-period encoder, which is responsible for capturing the temporal dependency between different phases within a period. Inter-period tokens are the input for another encoder, called inter-period encoder, which has the ability to capture the global temporal dependency among different periods.
- **Accurate Predictors**: Accurate predictors consists of multiple single-period predictors, and each single-period predictor is a multi-layer perceptron or a fully connected layer.

![model_structure_resize.jpg](pic/model_structure_resize.jpg)

# Get Start
1.Install Python>=3.9 and the requirements

2.Download data. You can obtain all the benchmark datastes from [[Autoformer](https://github.com/thuml/Autoformer)] or [[Informer](https://github.com/zhouhaoyi/Informer2020)].

3.If you want to replicate the experiments related to PerioDformer in the paper, you can directly run the scripts, which correspond to different experimental results in different folders as described in the paper.

```bash
# Table 2 in chapter 4.2 Long-term time series forecasting.
bash scripts/Long_series forecasting/XXX,sh
# Table 3 in chapter 4.2 Different period in periodic division.
bash scripts/Diffierent_C/XXX.sh
# Figure2 in chapter 4.2 Efficient channel-independence.
bash scripts/Efficient_channel_independence/XXX.sh
# Figure5 in chapter 4.3 Increasing look-back window.
bash scripts/Increasing_look_back_window/XXX.sh
# Table 7 in chapter 4.4 Different noise.
bash scripts/Different_noise/XXX.sh
# Table 8 in chapter 4.4 Different seeds.
bash scripts/Different_seeds/XXX.sh
# Table 9 in chapter 4.4 The impact of instance normalization.
bash scripts/No_norm/XXX.sh
```

4.If you want to train this model yourself, please change the default dataset and parameters in run_longExp.py within your scripts.

- For datasets with fewer variables, such as ETTh1, ETTh2, ETTm1, ETTm2, or similar, we recommend setting the search space as follows: w ∈ {1, 2, 3, 4, 5, 6, 7, 8} and mlp_num ∈ {1, 2, 3, 4, 5} to achieve better forecasting performance. For datasets with more variables, we suggest using the default values of w=1 and mlp=1.
- With the variation of C, the input tokens for the encoder will also undergo significant changes, indicating that we need to adjust two internal hyperparameters of the encoder, namely d-model and d-ff, to achieve better predictive performance.



# Main Results

We experiment on six benchmarks, covering five main-stream applications. 
Compared with other forecasters, PerioDformer achieves state-of-the-art performance at forecasting both high-dimensional and low-dimensional time series. Specifically, PerioDformer exhibits an average MSE and MAE reduction 4.898\% and 2.597\% for ETTh1.
Additionally, PerioDformer tends to perform better with longer prediction length, achieving mostly optimal results when the prediction length is 720 and 960.

![main_results_resize.jpg](pic/main_results_resize.jpg)

# Baselines

We will keep adding series forecasting models to expand this repo:

- [x] TimeMixer
- [x] iTransformer
- [x] PatchTST
- [x] RLinear
- [x] DLinear
- [x] TimesNet
- [x] Crossformer
- [x] FiLM

# Data Processing

We have provided several Python scripts for processing data. Please remember to modify the file paths inside the scripts before using them.

```bash
# The method of adding noise in Chapter 4.4 "Different Noise".
python data_processing/add_noise.py
# Extract the MSE and MAE values from the obtained result.txt file into a CSV file.
python data_processing/procese_mse_mae.py
```



# Contact

If you have any questions or want to use the code, please contact 2458488571@mail.dlut.edu.cn.








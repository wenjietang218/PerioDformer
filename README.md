# PerioDformer
We propose periodic Division Enhanced Transformer (PeriD
former) for time series forecasting, which transforming input series into input
 tokens with more semantic meaning.
Experimental data show that PeriDformer achieves state-of-the
art performance on six challenging real-world datasets.

# Get Start
1.Install Python>=3.9 and the requirements

2.Download data. You can obtain all the benchmark datastes from [[Autoformer](https://github.com/thuml/Autoformer)] or [[Informer](https://github.com/zhouhaoyi/Informer2020)].

3.If you want to replicate the experiments related to PerioDformer in the paper, you can directly run the scripts, which correspond to different experimental results in different folders as described in the paper.
<br />\scripts\Long_series_forecasting corresponds to Table1 
<br />\scripts\Different_C corresponds to Table2
<br />\scripts\Increasing_look_back_winodw corresponds to Figure2 \ Table11
<br />\scripts\Efficient_channel_independence corresponds to Figure3 \ Table12
<br />\scripts\Different_random corresponds to Table5
<br />\scripts\No_norm corresponds to Table9
<br />Attention: For the long forecasting results of ETTh1, we modified some hyperparameters. In Table1, with an input length of 336, the MSE and MAE values for output lengths of 96, 192, 336, 720, and 960 are changes as follows:

original:![img_1](https://github.com/wenjietang218/PerioDformer/assets/165779007/9bc89a15-e950-462d-bee5-757667af25d1) new:![img_2](https://github.com/wenjietang218/PerioDformer/assets/165779007/6a2193ca-c07d-4a6b-a0b3-3df594fa75a1)



4.If you want to train this model yourself, please change the default dataset and parameters in run_longExp.py within your scripts.
<br />For datasets with fewer variables, such as ETTh1, ETTh2, ETTm1, ETTm2, or similar, we recommend setting the search space as follows: w ∈ {1, 2, 3, 4, 5, 6, 7, 8} and mlp_num ∈ {1, 2, 3, 4, 5} to achieve better forecasting performance. For datasets with more variables, we suggest using the default values of w=1 and mlp=1.
<br />With the variation of C, the input tokens for the encoder will also undergo significant changes, indicating that we need to adjust two internal hyperparameters of the encoder, namely d-model and d-ff, to achieve better predictive performance.



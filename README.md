# Noisy ECG classification
### Submitted by: Idan David Or Lavi 



## Data

The data was taken from [CINC 2011](https://physionet.org/content/challenge-2011/1.0.0/#files-panel)
and [CINC 2017](https://physionet.org/content/challenge-2017/1.0.0/) datasets.

Some processing decisions:

> The data is cut to 10 second samples, respecting the sampling frequencies. 
>
> some of the data was lost in this process (in samples with length indivisible by 10 seconds). 
> this can be dealt with in more ways to avoid such data loss, and it's also part of the model design questions.
> 
> the data is then downsampled to 100Hz
> 


# Modeling and predicting
## The 2 colab notebooks are where the work happens

> in the `Data_analysis_and_modeling_acculine.ipynb` notebook you can find the stage 1 related work, 
post data processing (we start with train and test sets that are mostly ready to play nicely)
>
> some data processing is still preformed in this notebook, but it's modeling related (normalization) so it's more suitable to be part of the ML pipeline

> in the `Using_The_model.ipynb` notebook we have stage 2 code and explanations



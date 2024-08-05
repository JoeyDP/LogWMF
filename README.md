# The Role of Unknown Interactions in Implicit Matrix Factorization --- A Probabilistic View

Based on https://github.com/google-research/google-research/tree/master/ials.


## VAE benchmarks

This follows the evaluation protocol and uses the datasets from
[Liang et al., Variational Autoencoders for Collaborative Filtering, WWW '18](https://dl.acm.org/doi/10.1145/3178876.3186150).

### Instructions

1) Install packages `pip install -r requirements.txt`

2) Compile the code

- Download [Eigen](https://eigen.tuxfamily.org/):

```
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip
unzip eigen-3.3.9.zip
```
- Create the subdirectories

```
mkdir lib
mkdir bin
```
- Compile the binaries

```
make all
```

3) Download and process the data

```
python generate_data.py --output_dir ./
```

This will generate two sub-directories `ml-20m` and `msd` corresponding respectively to the data sets [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) and the [Million Song Data](http://millionsongdataset.com/tasteprofile/).

Note: this code is adapted from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb which requires a Python 3 runtime.


4) Run the training and evaluation code. Example usage:

**MovieLens 20M (ML20M)**

```
./bin/ialspp_main --train_data ml-20m/train.csv --test_train_data ml-20m/test_tr.csv \
  --test_test_data ml-20m/test_te.csv --embedding_dim 256 --stddev 0.1 \
  --regularization 0.003 --regularization_exp 1.0 --unobserved_weight 0.1 \
  --epochs 16 --block_size 128 --eval_during_training 0
```

**Million Song Data (MSD)**

```
./bin/ialspp_main --train_data msd/train.csv --test_train_data msd/test_tr.csv \
  --test_test_data msd/test_te.csv --embedding_dim 256 --stddev 0.1 \
  --regularization 0.002 --regularization_exp 1.0 --unobserved_weight 0.02 \
  --epochs 16 --block_size 128 --eval_during_training 0
```

Setting the flag `--eval_during_training` to 1 will run evaluation after each epoch.

## Reproducibility

Logs of runs can be found here: https://drive.google.com/drive/folders/1ElGp6pfxQxUYv2QWvBD-u0AUWIIuaKP_?usp=drive_link.

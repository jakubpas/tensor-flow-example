# Tensor flow example of neural network with text vectorized layer

The purpose of the application is to predict negative music reviews. There are two python scripts: one for nn training 
and one for running actual predictions. 
The training and tests sets are based on 6000 music reviews from the website https://www.darkplanet.pl and
their respective scores. Reviews are scored from 1 to 5 where score =<3 is considered negative. 
A vectorized layer is adapted to take standardized strings of given dictionary size. 
The Best results were obtained with 30 epochs and dropout 0.2. Training data set is split to validation
set with default ratio 0.3.  

## How to use

1. Unzip training set and test set:
```shell
tar -xzf data.tgz
```
2. Train model on set
```shell
./train.py
```
2. Evaluate predictions
```shell
./predict.py data/test/pos/38896.txt 
./predict.py data/test/neg/52449.txt
```

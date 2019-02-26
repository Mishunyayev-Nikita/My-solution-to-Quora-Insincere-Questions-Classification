## My solution Quora Insincere Questions Classification 2018-2019

I with my friend ranked 677th (TOP 17%) in the [Quora Insincere Questions Classification on Kaggle platform](https://www.kaggle.com/c/quora-insincere-questions-classification/leaderboard). We tried many neural network architectures in this competition, but we achieved our two best results by blending two models: rnn and rcnn. It was an interesting competition, primarily because I had never used recurrent and convolutional neural networks before. Also, the public leaderboard is calculated with approximately 15% of the test data and private with 85%, which led to a large shake-up and to the fact that different seed gave different results, even on a private leaderboard. We flew far down, but it was a great experience, from which I learned a lot of new things.

## In this repository you can find:
* `rnn.py` - **Reccurent neural network** with Attention, bidirectional LSTM and bidirectional GRU layers
* `rcnn.py` - **Recurrent convolutional neural network** with bidirectional LSTM, bidirectional GRU, Conv1d and pooling layers (average and max)
* `utils.py` - **Cyclic learning rate** and **Attention** code, that was used in NNs
* `training.ipynb` - notebook with data preprocessing (text cleaning, getting meta features) and training two NNs
* `check_vocab.py` - a script containing functions to track the learning vocabulary (which scans all of our text and counts the occurrence of words contained), and checks the intersection between our vocabulary and the embeddings (it will output a list of out of vocabulary words that we can use to improve our preprocessing)

## Teammate:
- [Andrew Lukyanenko](https://github.com/Erlemar/)

# Movie Recommender Service

The project attempts to recommend movies using few different approaches ( Content-based, Collaborative filtering and Deep Learning). It uses information from [MovieLens100K](https://grouplens.org/datasets/movielens/) and [TMDB5K](https://www.kaggle.com/tmdb/tmdb-movie-metadata) datasets.

## Installation 

```bash
pip install -r requirements.txt
```
## Datasets
As already mentioned, two datasets were used for system implementation. [MovieLens](https://grouplens.org/datasets/movielens/), for collaborative-filtering methods, and [TMDB](https://www.kaggle.com/tmdb/tmdb-movie-metadata) for content-based method.

Just worth mentioning, the data is split in 80:20 ratio, to training/test datasets. That ratio is used among all trained models.

### MovieLens
[MovieLens](https://grouplens.org/datasets/movielens/) is one of the most popular movie review datasets. The whole dataset contains 25 million ratings and is around 250MB in size. For testing and development purposes, a smaller sample containing 100.000 ratings is used.

In this dataset, every user is represented with an id and has at least 20 movie reviews. Every revie has a rating which is made on a 5-star scale, with half star increments (0.5 stars - 5.0 stars). For the movies, they are represented using id, title and genres.

This dataset is used in all collaborative-filtering based methods. In those cases genres were not used and are simply ignored columns of dataset.

### TMDB
[TMDB](https://www.kaggle.com/tmdb/tmdb-movie-metadata) dataset contains a bit smaller portion of movies (around 5000), but each movie is represented with additional properties such as *cast*, *director*, *overview*, etc. In that form, it was more appropriate for content-based approach. Since both datasets contain information about movie title, it was possible to find pairs in both datasets.

Dataset contains a lot of properties that are not neccessary affecting recommendations. For example *runtime*, representing length of a movie.

During preprocessing data was reshaped, which will be discussed later on.

## Content based approach

As the name suggests, *content-based* method is based on item (movie) metadata. Goal is to, somehow, find similarity metric between items and recommend items similar to ones the user already liked. The obvious drawback of this approach is not taking other users in consideration. The good side is that it doesn't have a *cold start* problem.

### Data preprocessing

As mentioned before, some of the properties in [TMDB](https://www.kaggle.com/tmdb/tmdb-movie-metadata) are not be that interesting, in terms of movie recommender. 

What seems to affect the recommendations the most are following attributes:
* keywords - list of keywords
* cast - top 3 actors
* director - movie director
* genres - movie genres
* overview - short movie description

### Implementation

After data preprocessing stop words were removed. After that, all of the extracted attributes, except the overview, were concatenated into a string. Such strings are compared using *cosine similarity* metric and *similarity matrix* is created. 

Using the *similarity matrix* we can compute similar movies to any give one. This, combined with user watching history, results in a simple *content-based* movie recommender.

For the movie overview, as it consists of larger, not structured portion of text, TFIDF is used. That way we can calculate *cosine similarity* on vectors.

Implementation can be seen in [this file](Content%20Based%20-%20Credits,%20Genres%20and%20Keywords.ipynb)


## Collaborative filtering approach - surprise
First approach to collaborative filtering is dependant on [surprise](https://github.com/nicolashug/Surprise) library, built on top of *scikit*, with purpose of simple development of python recommendation engines with collaborative filtering approach.

It provides various ready-to-use prediction algorithms such as baseline algorithms, neighborhood methods, matrix factorization-based ( SVD, PMF, SVD++, NMF), and many others. Also, various similarity measures (cosine, MSD, pearson...) are built-in.

After testing out various algorithms on *MovieLens* dataset, results are presented in the table below.

| Algorithm                                                                                                                              |   RMSE |   MAE | Time    |
|:---------------------------------------------------------------------------------------------------------------------------------------|-------:|------:|:--------|
| [Baseline](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)   |  1.513 | 1.214 | 0:00:13 |
| [SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)      |  0.936 | 0.738 | 0:06:71 |
| [k-NN](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)                        |  0.977 | 0.770 | 0:00:28 |
| [NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)      |  0.964 | 0.758 | 0:04:28 |

Looking at the results, SVD gives the best results, but is also the slowest one for training.

Implementation can be seen in [this file](Collaborative%20filtering%20-%20surprise.ipynb)

## Collaborative filtering approach - fastai 

Second approach to collaborative filtering is dependant on [fastai](https://docs.fast.ai/collab.html) library, built on to of *pytorch*, with the same purpose as *surprise*.
It offers two models, *EmbeddingDotBias* and *EmbeddingNN*, where later one is using a deeper Neural Network, hence it's slower but far more accurate.

Training it out on the same dataset, it resulted in a bit worse results in comparison to *surprise* approach. *MSE* loss on test data is 0.869.

The reason behind this is, probably, insuficient data for such a deep model. If we were using 25M version of dataset, results would definitely be more accurate.

Implementation can be seen in [this file](Collaborative%20filtering%20-%20fastai.ipynb)

## Collaborative filtering approach - keras

Final approach to deep learning collaborative filtering approach is dependant on [keras](https://keras.io/) and [tensorflow](https://www.tensorflow.org/) libraries.
Using given library a model with 140.000 parameters is created and used as *LFM* (*Latent factor model*).

Neural network summary:
```
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_5 (InputLayer)            [(None, 1)]          0                                            
__________________________________________________________________________________________________
input_6 (InputLayer)            [(None, 1)]          0                                            
__________________________________________________________________________________________________
embedding_4 (Embedding)         (None, 1, 50)        47200       input_5[0][0]                    
__________________________________________________________________________________________________
embedding_5 (Embedding)         (None, 1, 50)        84150       input_6[0][0]                    
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 50)           0           embedding_4[0][0]                
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 50)           0           embedding_5[0][0]                
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 100)          0           flatten_4[0][0]                  
                                                                 flatten_5[0][0]                  
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 100)          10100       concatenate_2[0][0]              
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 100)          0           dense_4[0][0]                    
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)            101         dropout_2[0][0]                  
==================================================================================================
Total params: 141,551
Trainable params: 141,551
Non-trainable params: 0
__________________________________________________________________________________________________
```

After 20 epochs of training, loss on test data has come down to 0.863, which is a tiny bit better than *fastai* approach.
The issue is same, insufficient data. If we worked with 25M dataset version, results would definitely be better, but so would training time grow.

Implementation can be seen in [this file](Collaborative%20filtering%20-%20Keras.ipynb)

## Models summary

| Approach                                                                                                                               |   RMSE |   MAE | Time    |
|:---------------------------------------------------------------------------------------------------------------------------------------|-------:|------:|:--------|
| SVD + surprise                                                                                                                         |  0.936 | 0.738 | 0:06:71 |
| FastAI EmbeddingNN                                                                                                                     |  1.002 | 0.869 | 0:00:28 |
| Keras neural network                                                                                                                   |  0.974 | 0.863 | 0:04:28 |

## Hybrid model idea
Having in mind that we have three different approaches to the same problem, it would be meaningful to combine the results into single recommendation.
The simplest and relatively effective way would be combining all the results together and giving them score, either generated as an approximated rating (by neural networks), or by number of occurences with content-based approach. Later, we can use that score to implement a weighted random selection of recommendations.

## Live usage
System is used in a movie streaming service [Cinema](https://nenad-misic.github.io/cinema). On homepage there are movie suggestions based on user usage history.

All models are integrated into *Flask* application and deployed to *Heroku*. Information about users is stored inside *Mongo* database, on cloud, using *Mongo atlas*.
Together with datasets, data from database is fetched and models are retrained from time to time (for now, there is an endpoint for manual invocation of such procedure).
Backend application can be seen on this [repo](https://github.com/nenad-misic/movie-recommender-service).

Frontend application uses various APIs and services for data collection/scraping and is all present on this [repo](https://github.com/nenad-misic/cinema).

## Author
Nenad Mišić, R2-19/2020

# Word embedder

This session aims at embedding the words resulting from the tokenization of the TripAdvisor's reviews. Preliminary ML techniques are applied to classify the ratings in order to assess the efficiency of the different embedding techniques.


## Architecture

The embedder has the following subfolders:
* ``` src ``` : contains all the ```.py``` files.
* ```notebooks ```: contains Jupyter notebooks to provide some visualisation and a Spark implementation of the lsi embedding in order to perform it with large datasets. 
* ``` trained_models ```: contains the ```.h5``` files to import the pretrained models without needing to run the heavy training optimization process.
* ``` embedded_data ```: contains the embedded reviews and their associated rating vectors.


## Run from Command Line

From the ```src``` folder:
```
python3 main.py --embed=True --classify=True
```

Usage:
* --embed [bool]: If set to True, the words embedding will be conducted using the lsi, word2vec and fasttext techniques. /!\ These processes are time and RAM consuming. See the **Spark implementation** section below. If set to False, the files will be read from the ```../embedded_data``` folder. False by default.
* --classify [bool]: If set to True, Deep Neural Networks will be trained in order to predict the rating of the review from its embedding version. Ignored if set to False. False by default.


## Spark implementation

In order to avoid the RAM issue of conducting embedding techniques on large datasets, we implemented a Spark version of the lsi embedding. If ran on a distributed cluster, this code enables to parallelize the RAM consuming operation (TruncatedSVD) on several machines. 
This code is accessible in the ```notebooks/svd_spark.ipynb ``` notebook and in the ```src/src/spark_lsi.py ``` file.


## Visualize the outcome of Deep Neural Techniques performed on top of the different embedding techniques

WIP + Colab notebook


## Data Available in the repository

The folder ``` embedded_data ``` contains the output from running ``` python3 main.py --embed ``` to process the tokenized data available in the folder ``` ../cleaner/cleaned_data/ ```.
The ``` embedded_data ``` folder contains 4 subfolders corresponding to the 4 embedding techniques that we performed:
* ``` embedded_data/lsi ```
* ``` embedded_data/spark_lsi ```: a parallelized version of LSI supported by Spark
* ``` embedded_data/word2vec ```
* ``` embedded_data/fasttext ```

Each subfolder contains the 5 files. For example, the ```embedded_data/spark_lsi ```:
* ```spark_lsi.csv```
A training set corresponding to 80% of the reviews from the ```spark_lsi.csv``` file. The different labels (corresponding to ratings from 1 to 5) are equally balanced. The target is one-hot-encoded as vectors of size 5. 
* ```X_train_spark_lsi.csv```
* ```y_train_spark_lsi.npy```
A testing set corresponding to 20% of the reviews from the ```spark_lsi.csv``` file. The different labels (corresponding to ratings from 1 to 5) are equally balanced. The target is one-hot-encoded as vectors of size 5.
* ```X_test_spark_lsi.csv```
* ```y_test_lsi.npy``` 
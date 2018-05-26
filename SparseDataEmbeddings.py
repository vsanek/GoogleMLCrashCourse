#---------------------------------------------------------------------------------------
# From Google Crash Course "Sparse Data and Embeddings"
# https://colab.research.google.com/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb

# Copyright 2017 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------------------
# Learning Objectives:
# Convert movie-review string data to a sparse feature vector
# Implement a sentiment-analysis linear model using a sparse feature vector
# Implement a sentiment-analysis DNN model using an embedding that projects data into two dimensions
# Visualize the embedding to see what the model has learned about the relationships between words

# Setup
print("Sparse Data and Embeddings START")
import collections
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)

train_path = 'train.tfrecord'
test_path = 'test.tfrecord'

def main():
    # ---------------------------------------------------------------------------------------
    # Building the Input Pipeline
    def _parse_function(record):
      """Extracts features and labels.
  
      Args:
        record: File path to a TFRecord file    
      Returns:
        A `tuple` `(labels, features)`:
          features: A dict of tensors representing the features
          labels: A tensor with the corresponding labels.
      """
      features = {
        "terms": tf.VarLenFeature(dtype=tf.string), # terms are strings of varying lengths
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32) # labels are 0 or 1
      }
  
      parsed_features = tf.parse_single_example(record, features)
  
      terms = parsed_features['terms'].values
      labels = parsed_features['labels']

      return  {'terms':terms}, labels

    # Create the Dataset object.
    ds = tf.data.TFRecordDataset(train_path)
    # Map features and labels with the parse function.
    ds = ds.map(_parse_function)

    display.display(ds)

    # Retrieve the first example
    n = ds.make_one_shot_iterator().get_next()
    sess = tf.Session()
    display.display(sess.run(n))


    # Create an input_fn that parses the tf.Examples from the given files,
    # and split them into features and targets.
    def _input_fn(input_filenames, num_epochs=None, shuffle=True):
  
      # Same code as above; create a dataset and map features and labels.
      ds = tf.data.TFRecordDataset(input_filenames)
      ds = ds.map(_parse_function)

      if shuffle:
        ds = ds.shuffle(10000)

      # Our feature data is variable-length, so we pad and batch
      # each field of the dataset structure to whatever size is necessary.
      ds = ds.padded_batch(25, ds.output_shapes)
  
      ds = ds.repeat(num_epochs)

  
      # Return the next batch of data.
      features, labels = ds.make_one_shot_iterator().get_next()
      return features, labels


    #---------------------------------------------------------------------------------------
    # Use a Linear Model with Sparse Inputs and an Explicit Vocabulary
    print("Use a Linear Model with Sparse Inputs and an Explicit Vocabulary")

    # 50 informative terms that compose our model vocabulary 
    informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                         "excellent", "poor", "boring", "awful", "terrible",
                         "definitely", "perfect", "liked", "worse", "waste",
                         "entertaining", "loved", "unfortunately", "amazing",
                         "enjoyed", "favorite", "horrible", "brilliant", "highly",
                         "simple", "annoying", "today", "hilarious", "enjoyable",
                         "dull", "fantastic", "poorly", "fails", "disappointing",
                         "disappointment", "not", "him", "her", "good", "time",
                         "?", ".", "!", "movie", "film", "action", "comedy",
                         "drama", "family")

    terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)



    my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    feature_columns = [ terms_feature_column ]


    classifier = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      optimizer=my_optimizer,
    )

    classifier.train(
      input_fn=lambda: _input_fn([train_path]),
      steps=1000)

    evaluation_metrics = classifier.evaluate(
      input_fn=lambda: _input_fn([train_path]),
      steps=1000)
    print("Training set metrics:")
    for m in evaluation_metrics:
      print(m, evaluation_metrics[m])
    print("---")

    evaluation_metrics = classifier.evaluate(
      input_fn=lambda: _input_fn([test_path]),
      steps=1000)

    print("Test set metrics:")
    for m in evaluation_metrics:
      print(m, evaluation_metrics[m])
    print("---")


    #---------------------------------------------------------------------------------------
    # Use a Deep Neural Network (DNN) Model
    print("Use a Deep Neural Network (DNN) Model")

    ##################### Here's what we changed ##################################
    classifier = tf.estimator.DNNClassifier(                                      #
      feature_columns=[tf.feature_column.indicator_column(terms_feature_column)], #
      hidden_units=[20,20],                                                       #
      optimizer=my_optimizer,                                                     #
    )                                                                             #

    try:
      classifier.train(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000)

      evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([train_path]),
        steps=1)
      print("Training set metrics:")
      for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
      print("---")

      evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([test_path]),
        steps=1)

      print("Test set metrics:")
      for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
      print("---")
    except ValueError as err:
      print(err)


    #---------------------------------------------------------------------------------------
    # Use an Embedding with a DNN Model
    print("Use an Embedding with a DNN Model")

    terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
    feature_columns = [ terms_embedding_column ]

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[20,20],
      optimizer=my_optimizer
    )


    classifier.train(
      input_fn=lambda: _input_fn([train_path]),
      steps=1000)

    evaluation_metrics = classifier.evaluate(
      input_fn=lambda: _input_fn([train_path]),
      steps=1000)
    print("Training set metrics:")
    for m in evaluation_metrics:
      print(m, evaluation_metrics[m])
    print("---")

    evaluation_metrics = classifier.evaluate(
      input_fn=lambda: _input_fn([test_path]),
      steps=1000)

    print("Test set metrics:")
    for m in evaluation_metrics:
      print(m, evaluation_metrics[m])
    print("---")

    #---------------------------------------------------------------------------------------
    # Convince yourself there's actually an embedding in there
    print("Convince yourself there's actually an embedding in there")

    display.display(classifier.get_variable_names())
    display.display(classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape)


    #---------------------------------------------------------------------------------------
    # Examine the Embedding
    print("Examine the Embedding")

    embedding_matrix = classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

    for term_index in range(len(informative_terms)):
      # Create a one-hot encoding for our term. It has 0s everywhere, except for
      # a single 1 in the coordinate that corresponds to that term.
      term_vector = np.zeros(len(informative_terms))
      term_vector[term_index] = 1
      # We'll now project that one-hot vector into the embedding space.
      embedding_xy = np.matmul(term_vector, embedding_matrix)
      plt.text(embedding_xy[0],
               embedding_xy[1],
               informative_terms[term_index])

    # Do a little setup to make sure the plot displays nicely.
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    plt.show()

main()

print("Sparse Data and Embeddings END")

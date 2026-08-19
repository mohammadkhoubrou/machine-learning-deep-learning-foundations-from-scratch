# Data Preparation

Before setting up a machine learning or deep learning model, we should decide on how we are going to use the data. One common practice among the developers is separating the dataset into three different groups; training set, dev(development) set, and test set. The reason for this division is to train the model on an adequate amount of data and evaluate it on an unseen set of data(not used during the training). What exactly are each of these sets for?

## Training Set

As you saw in the previous sections, training set is used to train the model. In the fitting process, by feeding the data to the structure, the model purposes a prediction, evaluates the prediction error, and alters the value of parameters using an optimization algorithm such as gradient descent to reduce the cost. 

Consider a training a model to detect plant diseases from a set of plant images. The dataset for this task contains images of plants and their corresponding labels. The model is trying to figure out the relationship between the images and the labels.

## Dev (Development) Set

The development set is also known as the validation set. It is used to evaluate the model during the development process.

Weights and biases are not altered during this phase. Instead, we use dev set to make decisions about the model itself, such as choosing the architecture, learning rate, regularization strength, or other hyper parameters. This allows us to compare different models and configurations without using the test set.

## Test Set

Finally, to evaluate the final version of the model, after the adjustments were made during the development level, we use the test set. The purpose of this evaluation is to see how well the model performs on unseen data. For some problems, we might need to apply the model in real world to make predictions based on the real world data. Such an evaluation can give us a pretty well understanding of how the model will react to different scenarios. 
Therefore, the general process can be described as,

```
Training set → learn the parameters
Development set → choose and evaluate the model
Test set → evaluate the final model
```
Data distribution varies based on some factors such as the task in hand, amount of data, and etc. In some cases you can divide the data into `80%` training set, `10%` dev set, and `10%` test test. Or even for very large amounts of data it is suggested to do a `98%`, `1%`, `1%` distribution.

$$
70% \rightarrow Training
$$

$$
15% \rightarrow Development
$$

$$
15% \rightarrow Test
$$



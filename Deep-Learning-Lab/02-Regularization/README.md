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

# Bias and Variance

To evaluate the trained model, and after the adjustments were made to model hyper parameters and structure, we will compare the model performance on training set vs dev set. In order to do that, we will use two concepts, *bias* and *variance* that'll us help get a better intuition of model performance. *High bias* occurs when the model fits neither the training set nor the dev set. Both sets return a high error. This circumstance is known as *underfitting*. An example of this situation is when our data is highly non-linear, however, our choice of model is a linear model. So the system doesn't have enough flexibility to learn the complicated trends in the data.

*High variance* describes the model performing pretty well in training set but not on the dev set. In other words, there are less errors on training set than dev set. The model has *overfit* the training data. Envision a model that has learnt the data points too perfectly that makes predictions almost as accurate as the training target values.

```
High bias → model is too simple → underfitting
High variance → model is too complex → overfitting
```

The goal is to find a model that flexibile enough to learn the important patterns without memorizing the training data.

# Regularization

One way to reduce the risk of overfitting is to prevent the parameters getting unreasonably large by introducing a penalty. Consider a polynomial model such as $\sum_{i=1}^{m} w_i x_i$. For any large value of w in this equation, any tiny change in input can trigger a great change in the model which results in poor generalization. Using a penalty to control the overfitting is called *regularization*. One common type of regularization is L2 regularization.

## L2 Regularization

In L2 regularization, also known as weight decay, we add the squared values of the weights to the cost function. For a neural network, the regularized cost function can be written as,

$$
J_{reg} =
J
+
\frac{\lambda}{2m}
\sum_{l=1}^{L}
\sum_{i}
\sum_{j}
\left(W_{ij}^{[l]}\right)^2
$$

where,

```
J : original cost function
J_reg : regularized cost function
λ : regularization parameter
m : number of training instances
W^[l] : weight matrix of layer l
L : number of layers
```

The bias terms are usually not included in the regularization term.
It is important to choose the correct value for $\lambda$. If $\lambda$ is too small, the regularization effect may not be sufficient to reduce overfitting. If it is too large, the model may become too constrained and underfit the data. The goal is not to reduce the weights to zero, but to make the relatively small.

### Frobenius Norm

Using the *forbenius*, we can express the same regularization term. The forbenius norm of matrix w is,

$$
|W|_F =
\sqrt{
\sum_i\sum_j W_{ij}^2
}
$$

Therefore,

$$
|W|_F^2 =
\sum_i\sum_j W_{ij}^2
$$

The cost function becomes, 

$$
J_{reg} =
J
+
\frac{\lambda}{2m}
\sum_{l=1}^{L}
|W^{[l]}|_F^2
$$

This is simply another way of writing the sum of the squared weights. When using L2 regularization, the derivative of the regularized cost with respect to a weight matrix becomes,

$$
dW^{[l]}_{reg} = 
dW^{[l]}
+
\frac{\lambda}{m}W^{[l]}
$$

The gradient descent update then becomes,

$$
W^{[l]}
:=
W^{[l]} -
\alpha
\left(
dW^{[l]}
+
\frac{\lambda}{m}W^{[l]}
\right)
$$

## Dropout Regularization

An other approach to reduce the overfitting is *dropout* method. In this method, we randomly remove some of the neurons in a layer, so the output becomes zero. In the next training iteration, other neurons are removed. Dropout prevents the model from relying too heavily on some neurons. In simpler words, in each iteration by removing some neurons, we temporarily make the network simpler. Dropout is only applied during training and not the development and predictions, the neurons work normally. It is also possible to apply this on the input layer.

### Inverted Dropout

One common implementation of dropout is called inverted dropout. It is designed to keep the expected value of the activations approximately unchanged after dropout is applied.
First, create a dropout mask.

$$
D^{[l]} =
\text{random values}
<
p
$$

where (p) is the probability that a neuron is kept. The mask contains either (1) or (0). For example,

$$
D =
\begin{bmatrix}
1 & 0 & 1 & 1
\end{bmatrix}
$$

means that the first, third, and fourth neurons are kept while the second neuron is dropped. We then apply the mask to the activation,

$$
A^{[l]} = 
A^{[l]}
\odot D^{[l]}
$$

$\odot$ represents element-wise multiplication.
Since the average activations are now smaller due to the deactivation of some nerons, we divide the average by the keep probability (P).

$$
A^{[l]} =
\frac{A^{[l]}\odot D^{[l]}}{p}
$$

For example, if the keep probability is,

$$
p=0.8
$$

then approximately (80%) of the neurons are kept, and the surviving activations are divided by (0.8). The expected value of the activation is therefore approximately preserved.

### Pseudo Code of Dropout

Dropout Function(A, keep_prob):

```
D = random_matrix_same_shape_as(A)

D = D < keep_prob

A = A * D

A = A / keep_prob

return A, D
```

During back propagation, the same dropout mask is used for the corresponding layer.

$$
dA^{[l]} =
\frac{dA^{[l]}\odot D^{[l]}}{p}
$$

The inactive neurons, do not effect the gradient descent in that iteration. 

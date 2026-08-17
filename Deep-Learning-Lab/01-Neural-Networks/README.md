# Neural Network
*Neural networks* are consisted of various layers of stacked *units* that take an input, perform calculations, and produce an output. These units are better known as ***neurons***.
Each neuron includes parameters that produce the output. These parameters are variable, and learning happens through altering the parameters to reduce the prediction error by
comparing the predictions to the desired results.
Simplest form of a neural network is a *single neuron neural network*.
<img align=center width="1536" height="1024" alt="neural networks" src="https://github.com/user-attachments/assets/956f765d-48a8-462f-9311-32ffc3393ef1" />

## Neural Networks for Supervised Learning
Turns out a neural network could meet the needs of the users and developers in different cases. Therefore, it was applied to various problems, but with some adjustments.
### Applications of NN
#### Standard NN
- Regression task. Housing price prediction(Real estate).
- Company's future revenue prediction.
- Whether an add will get clicked on based on user info and ad features.
#### **CNN**(Convolutional Neural Network)
- It's used in image classification tasks. For example, classifying images of different types of tumors (malignant or healthy).
- Object segmentation in images to locate the particles and shapes of them.
- For regression tasks. Whether the image shows a dog or not.
#### **RNN**(Recurrent Neural Network)
- Natural language processing(**NLP**). Fake news detection by processing and training a model on legitimate and false news.
- Speech recognition like voice operated applications.
- Machine translators.</br>

For each algorithm, a certain type of data is required. For instance, Housing price prediction, the data must include features of each training example. This type of data storage is known as *structured data*. An *unstructured data* however, is like image, text, or audio. No order is required.

## Neuron
Every neuron in a network is consisted of two back to back sections. After the data is fed to the neuron, it goes through a linear equation. Then the result is passed into the activation function.
<img align=center width="1536" height="1024" alt="single neuron" src="https://github.com/user-attachments/assets/d68c8a8e-6717-4bbe-9800-b66a6cff5c16" />
### linear combination
<p align= center>
$$
z=\sum_{i=1}^{m}w_{i}x_{j}+b
$$
</p>

- z : result of the linear combination
- $w_i$ : weights
- $x_j$ : training instances
- b : bias
- m : number of features


The dot product of the weights and the training examples is then added to a bias.</br>
- Every node(neuron) gets a copy of the training set. Now this set can be the whole dataset which is not common, unless the number of data points is very low. Otherwise, we usually use *batches* to train the data. So we divide the whole data set into multiple batches and in each forward propagation, we'll feed one of the batches as input and update the weights.


a = g(z)


```
a : activation of the neuron
g : activation function
z : linear combination
```

The activation function is responsible for transforming the linear combination before passing the result to the output. Different activation functions are used depending on the problem and the position of the neuron in the network.

### Activation Function

An activation function is a mathematical function applied to the output of the linear part of a neuron. Without an activation function, stacking multiple neurons and layers would still result in a linear transformation. Therefore, activation functions allow neural networks to learn more complex relationships.

#### ReLu(Rectified Linear Unit)

One commonly used activation function is *ReLU*, which stands for Rectified Linear Unit.

$g(z)=\max(0,z)$

Simply, ReLu returns zero for every negative output and the number itself in case it is 

$$
g(z) =
\begin{cases}
0 & z < 0 \\
z & z \geq 0
\end{cases}
$$

For instance, for a z = -3, the activation function returns 0. And for a z = 5, it returns 5 itself. Such a behavior allows the model to learn non-linear relationships between data points and generalize better.

The downside of using ReLu is that the derivative of this activation function, while the $z$ is negative, is zero. So we don't have a slope, thus we might encounter some problems with gradient descent. Instead, we can use *leaky ReLu*.

$$
f(z) =
\begin{cases}
z & \text{if } z \geq 0 \\
\alpha z & \text{if } z < 0
\end{cases}
$$


#### Sigmoid

Another common activation function is *sigmoid*. It is considered to be a more fitting activation function for classification tasks. The sigmoid function takes any real number and returns a value between (0) and (1). Therefore, its output can be interpreted as the estimated probability that the target belongs to the positive class.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

#### tanh(z)
This activation function is usually used in hidden layer of neural network structure. In some cases, the mean value of the data is important for us; Therefore, if the mean value is zero, it is appropriate to use ```tanh(z)``` as the activation function of hidden units.

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

The downside of these activation functions is that if the value of z is too small or too large, then it slows down.

<img align=center width="1254" height="1254" alt="activation-function-graphing" src="https://github.com/user-attachments/assets/bdc91137-3877-4ffe-bf87-25cda030fa41" />


## Binary Classification Using a Neural Network

As mentioned before, one of the applications of neural networks is solving binary classification problems. Such problems, require answering the question of "Which of the two group does this item belong to?". We usually represent the two classes using (0) and (1).

```
y = 0 : negative class
y = 1 : positive class
```

For example, if the network produces,

$$
\hat{y}=0.87
$$

we can interpret that the model estimates an (87%) probability that (y=1).
If a threshold of (0.5) is used, we can convert this probability into a class prediction.

$$
\hat{y} =
\begin{cases}
1 & \sigma(z) \geq 0.5 \\
0 & \sigma(z) < 0.5
\end{cases}
$$

## Logistic Regression

It turns out that a single neuron with a sigmoid activation function is mathematically equivalent to logistic regression.
First, the neuron calculates the linear combination,

$$
z = \sum_{j=1}^{n}w_jx_j+b
$$

Then, the result is passed through the sigmoid function,

$$
\hat{y} = \sigma(z)
$$

Therefore,

$$
\hat{y} = \frac{1}{1+e^{-(\vec{w}\cdot\vec{x}+b)}}
$$

The output $\hat{y}$ represents the estimated probability of (y=1) for the given input $\vec{x}$.

## Cost Function

In order to evaluate the predictions of our model, we need to compare the output with the labels set for each training instance. For binary classification, we use the logistic regression loss function.

$$
L^{(i)} =
-\left[
y^{(i)}\log(\hat{y}^{(i)})
+
(1-y^{(i)})\log(1-\hat{y}^{(i)})
\right]
$$

```
L^(i) : loss for training instance i
y^(i) : target value
ŷ^(i) : model's prediction
```

The loss function measures the prediction error. It calculates how wrong the model is for a single training example. If the prediction is close to the label, the loss is small. However, if the prediction is way off, the loss gets larger. The cost function simply aggregates the errors or losses of the whole training set into one value which becomes the overall performance measure of the model.

$$
J(\vec{w},b) =
-\frac{1}{m}
\sum_{i=1}^{m}
\left[
y^{(i)}\log(\hat{y}^{(i)})
+
(1-y^{(i)})\log(1-\hat{y}^{(i)})
\right]
$$

```
m : number of training instances
J(w,b) : cost function
```

The purpose of training the model is to minimize the value of the cost function by adjusting the parameters such as weights and bias.

## Gradient Descent

As you might remember from previous sections, our defined system learns by adjusting the parameters. In fact, we are looking for the best combination of the parameters together. How do we know that the model has learned? When it shows the least error or difference between the target value and model prediction.

By minimizing the cost function, we try to obtain parameters that produce more accurate predictions. However, this won't happen on its own. Thus, we need an algorithm to update the values automatically until the cost function reaches a minimum.

Gradient Descent is an algorithm used for this purpose. It uses derivatives of the cost function to determine how the parameters should be changed.

The parameters are updated using the following equations.

$$
w_i := w_i - \alpha \frac{\partial}{\partial w_i}J(\vec{w},b)
$$

$$
b := b - \alpha \frac{\partial}{\partial b}J(\vec{w},b)
$$

```
w_i : weight corresponding to feature i
b : bias
α : learning rate
J(w,b) : cost function
```

## Learning Rate

It is a hyper parameter. Hyper parameters are set from the beginning and are not directly learned by the model. The learning rate is a small positive number that determines how big or small the steps should be when updating the parameters.

Two cases emerge while trying to determine the value of a hyper parameter; First, if the value is too small, it may take more iterations to get to the optimal point. On the other hand, if it is too large, it might overshoot and never converge.

## Derivative

Derivatives are used to determine the rate of change of a function. For a simple function such as,

$$
f(x)=x^2
$$

by taking the derivative we get,

$$
f'(x)=2x
$$


The derivative tells us how quickly the output of the function changes when the input changes. For example, at (x=2),

$$
f'(2)=4
$$

This means that the function has a positive slope of (4) at that point.

## Partial Derivatives

The same principal is applied to multi variable functions. We take the derivative of a multi variable function with respect to a certain variable while considering the other variables as constants. Using partial derivatives, we can determine how the slope of the curve changes by altering that variable. For example,

$$
f(x,y)=x^2+y^2
$$

The partial derivative with respect to (x) is,

$$
\frac{\partial f}{\partial x}=2x
$$

and the partial derivative with respect to (y) is,

$$
\frac{\partial f}{\partial y}=2y
$$

## Gradient
In the case of neural networks, we adjust weights and bias to decrease the cost. Thus, we need to know how the cost function behaves while adjust each of these parameters. Gradient combines the partial derivatives of a function into a vector.

$$
\nabla f =
\begin{bmatrix}
\frac{\partial f}{\partial x}\
\frac{\partial f}{\partial y}
\end{bmatrix}
$$

This vector points to the direction of maximum increase and the negative gradient points to the maximum decrease. Using gradient we can find the direction where the cost function decreases instantly. However, the cost function doesn't directly depend on each parameter.

## Chain Rule

The parameters affect intermediate values, which eventually affect the final prediction and the cost. For example, consider the following sequence,

$$
x \rightarrow z \rightarrow a \rightarrow J
$$

where each value depends on the previous value. To determine how a change in (x) affects (J), we use the chain rule.

$$
\frac{dJ}{dx} =
\frac{dJ}{da}
\frac{da}{dz}
\frac{dz}{dx}
$$ 

The chain rule allows us to calculate the derivative of a function composed of multiple functions by multiplying the derivatives along the path.

## Computational Graph

A computational graph is a graphical representation of the sequence of operations used to calculate an output. Each node represents a variable or operation, and the connections represent the dependencies between them. For example, a simple neuron can be represented as,

$$
x,w,b \rightarrow z \rightarrow a \rightarrow J
$$

where,
$$
z = wx+b
$$

$$
a = g(z)
$$

and the cost function uses (a) and (y) to calculate (J).

The computational graph allows us to move forward through the operations to calculate the prediction and the cost. Then, we can move backward through the graph and use the chain rule to calculate the derivatives.

## Backpropagation

The process of calculating derivatives by moving backward through the computational graph is known as backpropagation. It applies the chain rule repeatedly to determine how the cost function changes with respect to each parameter.

For example, if the cost depends on the activation, which depends on (z), which depends on (w), we can calculate,

$$
\frac{\partial J}{\partial w} =
\frac{\partial J}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
$$

Similarly, for the bias,

$$
\frac{\partial J}{\partial b} =
\frac{\partial J}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial b}
$$

These derivatives tell us how much the cost function changes when the corresponding parameter changes. Gradient descent then uses these values to update the weights and bias.

For logistic regression, after applying the chain rule, the derivative of the cost function with respect to a weight can be simplified to,

$$
\frac{\partial J}{\partial w_j} =
\frac{1}{m}
\sum_{i=1}^{m}
(\hat{y}^{(i)}-y^{(i)})x_j^{(i)}
$$

and for the bias,

$$
\frac{\partial J}{\partial b} =
\frac{1}{m}
\sum_{i=1}^{m}
(\hat{y}^{(i)}-y^{(i)})
$$

These derivatives are then used in gradient descent to update the parameters and gradually reduce the cost function.

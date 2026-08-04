# Logistic Regression

## Binary Classification

As we have seen before, classification is one of the classic problems solved by supervised learning algorithms. In classification, instead of predicting a continuous value such as the price of a house, we are trying to determine which class an instance belongs to. One of the simplest forms of classification is **binary classification**, where there are only two possible classes.

For instance, consider a system that determines whether an email is spam or not spam. We can represent the two classes with `0` and `1`.

* `0` → not spam
* `1` → spam

The dataset contains instances with their attributes and a label assigned to each instance. The algorithm then learns the relationship between the attributes and the labels and uses what it has learnt to classify previously unseen instances.

However, there is an important problem with binary classification. Imagine that we have two groups of data points representing two different classes. What happens if some data points are located in the intersection of both classes? These instances make it difficult for the model to decide where the boundary between the two classes should be.

Another possible scenario is when some of the data points or training instances are far from the rest of the dataset. Such instances can strongly affect the behavior of the model in case of repetition. Therefore, we need an algorithm that can determine a more accurate boundary and produce a value to classify the unseen data into its suitable class. This is where **logistic regression** comes in.

## Logistic Regression
Logistic regression is mainly used for classification tasks. It is set to produce a value between `0` and `1` instead of a continues value, so that it can represent the probability of an instance belonging to a certain class. To achieve this, logistic regression uses the **sigmoid function**.

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

The sigmoid function takes any value as its input and maps it to a value between `0` and `1`.

For logistic regression, we first calculate:

$$
z = \vec{w} \cdot \vec{x} + b
$$

Then, we pass the result through the sigmoid function:

$$
f_{\vec{w},b}(\vec{x}) =
\frac{1}{1 + e^{-(\vec{w}\cdot\vec{x}+b)}}
$$

Now the output can be interpreted as a probability.

For example, if the model returns:

$$
f_{\vec{w},b}(\vec{x}) = 0.87
$$

we can say that the model estimates an `87%` probability that the instance belongs to class `1`.

We can then define a threshold. Usually, we use `0.5`.

$$
\hat{y} =
\begin{cases}
1 & \text{if } f_{\vec{w},b}(\vec{x}) \geq 0.5 \ 
0 & \text{if } f_{\vec{w},b}(\vec{x}) < 0.5
\end{cases}
$$

***Forward propagation*** is over. Now we have the predicted values for the first run through the data. Here is where the learning begins. The system must compare its predictions with the actual targets and alter the parameters to achieve a better accuracy. This is called the ***back propagation***, and it is consisted of various parts.

## Loss Function

As we discussed before, the difference between a prediction and the actual value is called **prediction error**.

However, the Mean Squared Error we used for linear regression is not the appropriate choice here. Since logistic regression produces probabilities, we need a loss function that properly evaluate these probabilities. For this purpose, we use the **logistic loss function**, also known as **log loss**.
<p align=center>
$L(f_{\vec{w},b}(\vec{x}),y) = -y\log\left(f_{\vec{w},b}(\vec{x})\right) - (1-y)\log\left(1-f_{\vec{w},b}(\vec{x})\right)$</p>

Let's see the following example.

If the actual label is `1`, then:

$y=1$

Therefore:

$L = -\log(f_{\vec{w},b}(\vec{x}))$

If the model predicts a value close to `1`, the loss becomes very small. But if it predicts a value close to `0`, the loss becomes very large. The same idea applies when the actual label is `0`. The model is rewarded for assigning a low probability to class `1` and heavily penalized when it is confidently wrong.

## Cost Function

The loss function measures how wrong the model's answer is for a **single training instance**. But we usually have many training instances. Therefore, we need to aggregate the losses from all training examples into one value. This is the purpose of the **cost function**.

<p align=center>
$J(\vec{w},b)= -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))\right]$</p>

Where:

* $f_{\vec{w},b}(\vec{x}^{(i)})$: model's prediction
* $y^{(i)}$: target value (label)
* $m$: number of training instances
* $J(\vec{w},b)$: the cost function

The purpose of training a model is to minimize the value of the cost function by adjusting the parameters such as weights and bias.

Now the question is: **How do we find the values of $\vec{w}$ and $b$ that minimize the cost function?**

For this purpose, we use **Gradient Descent**.

## Gradient Descent

As you might remember from previous sections, our defined system learns by adjusting the parameters. In fact, we are looking for the best combination of the parameters together. How do we know that the model has learnt? When it shows the least error or difference between the target value and model prediction.

So by minimizing the Cost function, we will get more accurate predictions. But we know that this won't happen on its own. Thus, we need an algorithm to update the values for us automatically until it hits the minimum point.

**Gradient Descent** is an algorithm used for this purpose.

The idea is the same as what we discussed with linear regression. We start with some initial values for our parameters and repeatedly update them in the direction that reduces the cost function.

The update rule for the weights is:

$$
w_j :=
w_j -
\alpha
\frac{\partial}{\partial w_j}
J(\vec{w},b)
$$

And for the bias:

$$
b :=
b -
\alpha
\frac{\partial}{\partial b}
J(\vec{w},b)
$$

For logistic regression, the derivatives simplify to:

$\frac{\partial}{\partial w_j}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m}\left(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}\right)x_j^{(i)}$

and:

$\frac{\partial}{\partial b}J(\vec{w},b) = \frac{1}{m}\sum_{i=1}^{m}\left(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}\right)$

As you can see, the general structure of the gradient is familiar. We calculate the difference between the model's prediction and the actual label, then use that difference to determine how the parameters should be adjusted.

### Pseudo Code of Gradient Descent

The process can be described in a few simple steps.

```text
GD Function(x, y, initial_w, initial_b, α, num_iterations):

w = initial_w
b = initial_b

for i in range(num_iterations):

    predictions = sigmoid(w · x + b)

    dj_w = (1/m) Σ(predictions - y)x
    dj_b = (1/m) Σ(predictions - y)

    w = w - α * dj_w
    b = b - α * dj_b

return w, b
```

The algorithm repeatedly calculates the predictions, calculates the gradient, and then updates the parameters. This process continues until we reach a point where the cost function is minimized or becomes sufficiently small.

# Optimization

As we experienced with different types of data, we came to realize that some problems can't be answered with a simple network architecture. If the data is too noisy, or the amount of training data is too large, we might need a larger network. However, as the amount of data or the seize of the architecture grows, so does the computational cost. Of course applying the system on better equipment could assist us with this issue, but that option is not always available. Therefore, several techniques were introduced to optimize and accelerate the learning process.

<img align=center width="1254" height="611" alt="image" src="https://github.com/user-attachments/assets/73965e79-7755-42cb-94dc-fe40af6836ba" />


## Data Normalization

Training dataset can contain attributes in different scales. For instance, some features might be binary (0 or 1), while others are on scale of a thousand. This can slow down the learning and parameters update because the cost function might be stretched in some directions. In order to prevent this problem, it is helpful if we normalize the data. One common method to do so, is *standardization*.

$$
\mu_j =
\frac{1}{m}
\sum_{i=1}^{m}
x_j^{(i)}
$$

```
μ_j : mean of feature j
m : number of training instances
x_j^(i) : value of feature j in training instance i
```

Then, we calculate the variance.

$$
\sigma_j^2 =
\frac{1}{m}
\sum_{i=1}^{m}
\left(x_j^{(i)}-\mu_j\right)^2
$$

The standard deviation is,

$$
\sigma_j=\sqrt{\sigma_j^2}
$$

Now, we can transform the original value into a new value centered around zero.

$$
x_{norm,j}^{(i)} =
\frac{x_j^{(i)}-\mu_j}{\sigma_j}
$$

After this transformation, the feature will have approximately zero mean and unit variance.
<img align=center width="1254" height="655" alt="image(2)" src="https://github.com/user-attachments/assets/ca3953d1-fb9c-4dff-913c-b34f99ad7d6d" />


## Parameter Initialization

The parameters of a neural network are usually assigned with random numbers. Simply because if the parameters of two various nodes have the same values, the output and the gradients of those nodes are equal. This problem is known as symmetric weights problem or network symmetry. Because of the same gradients, the parameters are together. In other word, two nodes learn the same features.
Envision a network in which the weights are initialized with identity matrix. Because of the following characteristic of the identity matrix,

$$
\begin{matrix}
W = I \\
IX = X
\end{matrix}
$$

Each node pass on an individual feature from X. This method preserves that feature before passing it to the activation function. If the weights are slightly larger than the identity matrix, gradients will get larger as they pass to the next layer in a large neural network. This phenomena is called *exploding gradient problem*.

$$
W \approx 1.1I
$$

On the other hand, if the weights are slightly smaller than the identity matrix, in the long run gradients become very small which is known as the *vanishing gradient problem*.

$$
W \approx 0.9I
$$

So it is necessary to initialize the weights in a way that the scale of the parameters don't change in training. One approach is using *Xavier initialization*.


Xavier Initialization

Xavier initialization also known as *Glorot initialization*, chooses the variance of the initial parameters based on the number of input and outputs of a layer.

$$
W^{[l]}
\sim
\mathcal{N}
\left(
0,
\frac{2}{n^{[l-1]}+n^{[l]}}
\right)
$$

```
n^[l-1] : number of input units
n^[l] : number of output units
```

This helps keep the scale of the activations and gradients from changing too much as they pass through the network. For example, if a layer has many input units, the initial weights should generally have a smaller variance. Otherwise, adding many weighted inputs can produce values that are unnecessarily large.

## Mini-Batch Gradient Descent

In the previous practices, we were using the whole dataset to make predictions and calculate the cost and gradients then update the parameters. This approach can be inefficient when a large number of training instances are stored in the dataset. Instead, we can divide the dataset into smaller groups called *mini-batches*. Then, the model makes predictions using a batch and updates the parameters. This method is known as *mini-batch gradient descent*. After all the mini-batches were used for training, a pass through the entire dataset is called an *epoch*. For example, if we choose a mini-batch size of 100 for a dataset consisted of 1000 instances, the dataset will be divided into 10 mini-batches. 

$$
m=1000
$$

and,

$$
B=100
$$

then,

$$
\frac{1000}{100}=10
$$


The algorithm is traced in following steps,

```
Shuffle the training set
Divide the shuffled data into mini-batches
Take the first mini-batch
Perform forward propagation
Calculate the cost
Perform back propagation
Calculate the gradients
Update the parameters
Repeat for the remaining mini-batches
```

### Mini-Batch Size

There's no value to choose as a mini-batch size. It usually depends on the size of the dataset and the hardware such as memory size. Mini-batch of a small size use less memory and update the parameters more frequently. However, the gradients of a larger mini-batch is more stable but require more memory. Most common sizes for a mini-batch are,

$$
32,\ 64,\ 128,\ 256
$$

### Shuffling and Partitioning
Usually the dataset is shuffled before creating mini-batches. The reason is that some datasets store the data sequentially. For example, the first section of a dataset might contain instances of one class, while the following section contains the instances of another class. So when dividing the dataset, some mini-batches might entirely be consisted of one class. Shuffling, changes the order of training examples before creating smaller groups.

$$
X =
[X^{{1}},X^{{2}},...,X^{{k}}]
$$

$$
Y =
[Y^{{1}},Y^{{2}},...,Y^{{k}}]
$$

where each $X^{{k}}$ represents one mini-batch and each $Y^{{k}}$ is the corresponding label. 

The input mini-batch and its corresponding labels must remain matched after shuffling and partitioning.

## Exponentially Weighted Averages

Another useful concept in optimization is the exponentially weighted average.

It is a method for calculating an average that gives more importance to recent values while gradually giving less importance to older values.

For a sequence $v_1,v_2,\ldots,v_t$, we can calculate,

$$
V_t =
\beta V_{t-1}
+
(1-\beta)v_t
$$

```
V_t : exponentially weighted average
v_t : current value
β : weighting parameter
```

If $\beta$ is close to (1), the average changes slowly and remembers more of the previous values. If it is smaller, the average reacts more quickly to new values. In neural networks, this idea can be used to create smoother estimates of gradients and their squared values.

## Gradient Descent with Momentum

Gradient descent with momentum is an optimization method that uses exponentially weighted averages of the gradients.
A clear illustration of this is considering a ball moving downhill. In addition to considering the current gradient descent, the algorithm remembers part of the previous direction too. First, the exponentially weighted average is calculated,

$$
V_{dW} =
\beta V_{dW}
+
(1-\beta)dW
$$

and,

$$
V_{db} =
\beta V_{db}
+
(1-\beta)db
$$

Then, the parameters are updated using these values.

$$
W := W-
\alpha V_{dW}
$$

$$
b := b-\alpha V_{db}
$$

The parameter $\beta$ controls how much information from previous gradients is retained. Momentum can help the optimization move faster in directions where the gradient consistently points in the same direction and reduce unnecessary oscillations.

## RMSProp

RMSProp stands for Root Mean Square Propagation. Instead of calculating an exponentially weighted average of the gradient itself, RMSProp keeps an exponentially weighted average of the squared gradients.

For the weights,

$$
S_{dW} =
\beta S_{dW}
+
(1-\beta)(dW)^2
$$

and for the bias,

$$
S_{db} =
\beta S_{db}
+
(1-\beta)(db)^2
$$

Then, the parameters are updated using,

$$
W := W -
\alpha
\frac{dW}
{\sqrt{S_{dW}}+\epsilon}
$$

$$
b := b -
\alpha
\frac{db}
{\sqrt{S_{db}}+\epsilon}
$$

where $\epsilon$ is a very small positive number added to prevent division by zero.

The main idea is to divide the gradient by an estimate of its recent magnitude. Therefore, parameters with consistently large gradients can receive smaller updates, while parameters with smaller gradients can receive relatively larger updates.

## Adam Optimization

Adam stands for Adaptive Moment Estimation. Combining the idea behind RMSProp and momentum, it keeps an exponentially weighted average of the gradients and the squared gradients.

It keeps an exponentially weighted average of the gradients and another exponentially weighted average of the squared gradients.

First, we calculate the first moment,

$$
V_{dW} =
\beta_1V_{dW}
+
(1-\beta_1)dW
$$

Then, we calculate the second moment,

$$
S_{dW} =
\beta_2S_{dW}
+
(1-\beta_2)(dW)^2
$$

The same procedure happens for the bias. Because these averages are initialized at zero, they can be biased toward zero during the first few iterations. Therefore, Adam uses bias correction.

$$
\hat{V}_{dW} =
\frac{V_{dW}}
{1-\beta_1^t}
$$

$$
\hat{S}_{dW} =
\frac{S_{dW}}
{1-\beta_2^t}
$$

Then, the parameters are updated,

$$
W := W -
\alpha
\frac{\hat{V}*{dW}}
{\sqrt{\hat{S}*{dW}}+\epsilon}
$$

The same procedure is used for (b).

The advantage of Adam is that it combines momentum, which keeps information about the direction of previous gradients, with adaptive scaling, which uses information about the magnitude of recent gradients.

## Activation Normalization

As the data passes through a neural network, the distribution of the activations can change from one layer to another. Therefore, we can normalize the activations inside the network as well. A common approach to this problem is *activation normalization*.

For a mini-batch, the activation normalization is,

$$
a^{(1)},a^{(2)},...,a^{(m)}
$$

First, we calculate the mean,

$$
\mu_B =
\frac{1}{m}
\sum_{i=1}^{m}a^{(i)}
$$

Then, we calculate the variance,

$$
\sigma_B^2 =
\frac{1}{m}
\sum_{i=1}^{m}
\left(a^{(i)}-\mu_B\right)^2
$$

The activation is then normalized,

$$
\tilde{a}^{(i)} =
\frac{a^{(i)}-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

However, forcing every activation to have exactly zero mean and unit variance may restrict what the network can learn. Therefore, Batch Normalization introduces two trainable parameters, (\gamma) and (\beta).

The final normalized activation is,

$$
a_{BN}^{(i)} =
\gamma\tilde{a}^{(i)}
+
\beta
$$

```
μ_B : mean of the mini-batch
σ_B² : variance of the mini-batch
ε : small positive number
γ : trainable scale parameter
β : trainable shift parameter
```

Therefore, the network can learn an appropriate scale and center for the normalized activation.

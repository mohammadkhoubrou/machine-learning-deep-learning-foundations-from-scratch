# Overfitting

So far, we have learned how to train a model on a dataset to learn the pattern hidden in the data. Sometimes, models tend to memorize the target values assigned to
a training instance. Thus, it underperforms while encountering unseen data. This phenomena is known as **overfitting**. It happens when the model is too familiar with
the data and start to learn unnecessary details and noises(data points that belong to a certain class, but they're far from their cluster). In this case, we can say
that the model doesn't **generalize**. There are several ways we can deal with overfitting.

### 1. Gathering More Data

By providing more training instances to the model, it can learn the general pattern way easier than before. Instead of relying on a few instances with countless attributes,
we can reduce the number of instances in a dataset, but prepare more data points for the model to examine and learn from.
However, this approach is not always possible. Most of the times, it is an expensive procedure for the project.

### 2. Feature Selection

Another way to reduce the sophistication of the model is to reduce the number of unnecessary attributes or features. If you see a complicated model that operates with attributes
that don't seem essential for the purpose of the system, you can simply remove the features to help the model generalize better.

### 3. Regularization

The third method is **Regularization**. The main idea behind this approach is to keep the size of the parameters smaller by simply adding an additional term to the 
cost function equation. This helps to prevent the model from getting unnecessarily complicated. 
For logistic regression, the regularized cost function becomes:

$J(\vec{w},b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) + (1-y^{(i)})\log\left(1-f_{\vec{w},b}(\vec{x}^{(i)})\right)\right]
+
\frac{\lambda}{2m}
\sum_{j=1}^{n}w_j^2$

The first part is the original logistic regression cost function. The second part is the **regularization term**:

$\frac{\lambda}{2m}
\sum_{j=1}^{n}w_j^2$

Here, $\lambda$ is a hyperparameter that determines how strongly we want to penalize large weights. Notice that we don't regularize the bias $b$. The regularization 
term is applied to the weights. If $\lambda$ is too small, the model may still overfit because the penalty is not strong enough. If $\lambda$ is too large, 
the model can become too simple and may *underfit* the training data (when the model can't even generalize on the training dataset). Therefore, we need to find 
a suitable value for $\lambda$.

## Gradient Descent for Regularized Logistic Regression

Now that we have changed our cost function, we also need to adjust the gradient descent equations. The derivative with respect to the weights now contains an 
additional regularization term:

$\frac{\partial}{\partial w_j}J(\vec{w},b)=
\frac{1}{m}
\sum_{i=1}^{m}
\left(
f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}
\right)x_j^{(i)}
+
\frac{\lambda}{m}w_j
$

However, the derivative with respect to the bias remains unchanged:

$\frac{\partial}{\partial b}J(\vec{w},b)=
\frac{1}{m}
\sum_{i=1}^{m}
\left(
f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}
\right)
$

Therefore, the update rules become:

$w_j :=
w_j -
\alpha
\left[
\frac{1}{m}
\sum_{i=1}^{m}
\left(
f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}
\right)x_j^{(i)}
+
\frac{\lambda}{m}w_j
\right]$

and:

$b :=
b -
\alpha
\left[
\frac{1}{m}
\sum_{i=1}^{m}
\left(
f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}
\right)
\right]$

As you can see, the main difference is the additional $\frac{\lambda}{m}w_j$ term in the weight update. This term pushes the weights toward smaller values 
during training. The overall process is still the same:

1. Initialize $\vec{w}$ and $b$.
2. Calculate the predictions using the sigmoid function.
3. Calculate the gradient.
4. Update $\vec{w}$ and $b$.
5. Repeat the process until the cost function reaches its minimum or becomes sufficiently small.

By combining logistic regression with gradient descent and regularization, we have a model that can learn a decision boundary for binary classification 
while reducing the possibility of overfitting.

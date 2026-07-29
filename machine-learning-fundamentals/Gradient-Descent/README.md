# Gradient Descent
As you might remember from previous sections, our defined system learns by adjusting the parameters. In fact, we are looking for the best combination of the parameters together.
How do we know that the model has learned? When it shows the least error or difference between the target value and model prediction. So by minimizing the Cost function, we will 
get more accurate predictions. But we know that this won't happen on its own. Thus, we need an algorithm to update the values for us automatically until it hits the minimum point.

*Gradient Descent* is an algorithm used for this purpose. Perhaps it would be helpful if we reviewed the concept of derivative and gradient before jumping into the algorithm itself.</br>

⭕ **If you're not familiar with derivatives and gradient, I strongly recommend a thorough study of these concepts.**

### Derivative
Derivatives are used to determine the rate of change of a function. For a simple functions such as f(x) = $x^{2}$, by taking the derivative we get f'(x) = 2x which is the rate for which the output of the function changes. As you can see the derivative is a function itself. We can also plot that one.
<p align=center>
<img width="370" height="490" alt="f(x) = x^2, f'(x) = 2x graph" src="https://github.com/user-attachments/assets/3b9696ad-1ab2-4d3e-8c74-a70f60b62d7b"/>
</p>
</br>
<p align=center>The pink line is the derivative of f(x). And the other line is the tangent line of the function at (1,1).</p></br>

### Partial Derivatives
Just as you witnessed, we can find the rate of change of a function at any given point. Moreover, we can apply the same principal to functions with more than one variable. *Partial derivatives* is the derivative of a function with respect to one of its variables. In simpler terms, we keep one of the variables and consider the rest as constants. This way we examine the behavior of the function for any change in that variable. It's like you're asking, if I move in the x direction and keep y fixed, how steep will the graph get? Or the other way around; If I move in y direction and keep x fixed, how steep will the graph get?
<p align=center>f(x,y)= $x^{2}$ + $y^{2}$ </p>
<p align=center>$\frac{df}{dx}$= 2x </p>
<p align=center>$\frac{df}{dy}$= 2y </p>

### Gradient ∇
In real life though, we don't only walk in direction of x or y. We are free to walk in any direction. The famous example to understand this concept is to imagine you're climbing a mountain. Then, you ask yourself, which direction should I climb to gain altitude quicker? A simple partial derivative can't solve this problem. Therefore, we use *gradient*.
Gradient combines both partial derivatives and displays it as a vector. Geometrically, it lies on the xy-plane and points toward the direction of maximum increase. The length of the vector specifies how steep the fastest climb is(largest value of z). The opposite direction of gradient is the steepest downhill direction and perpendicular routes experience no sudden change. In the following image you can grasp a better understanding of the gradient calculated different points on the graph.
<p align=center>
<img width="713" height="581" alt="Gradient vector field" src="https://github.com/user-attachments/assets/4d0eb5c0-3ee2-4fce-8a5d-9b8e1449c9e3" />
</p></br>
<p align=center>Gradient vector field (f∇) of f(x,y) = $x^{2}$ + $y^{2}$</p></br>

## Algorithm Explained
This algorithm can be described in three simple steps.
- Start with initializing w, b usually with 0
- After each run, alter w, b a bit to reduce J(w,b)
- Repeat the second step until you get near or to the minimum point
How do we update parameters?
<p align=center>w := w - α $\frac{d}{dw}$ J(w,b) x</p>
<p align=center>b := b - α $\frac{d}{db}$ J(w,b)</p>

### Learning Rate
It is a *hyper parameter*. Hyper parameters are set from the beginning and we don't change them throughout the training process. Learning rate is a small number usually between 0,1 that determines how big or small your steps should be when updating your parameters.

The derivative part determines the direction in which the parameter must be updated. If the value of derivative is +, it will take steps to the right of the curve, because the minimum is behind the current point. If the value is -, it will step right toward the minimum point.

- If the learning rate is too small, it will take more iterations to find the minimum.
- If it's too big, it will possible overshoot and will never find the minimum.
- To find the best learning rate, experiment with a small number of training instances with different numbers like 0.001, 0.01, 0.1.

### Logistic Regression Cost Function derivative

$\frac{d}{dw_j}$ J(w,b) = $\frac{1}{m}$ $\sum_{i=1}^{m}$ ($f_{w,b}(x^{i}) - y^{i}$) $x_{j}^{(i)}$

$\frac{d}{db_j}$ J(w,b) = $\frac{1}{m}$ $\sum_{i=1}^{m}$ ($f_{w,b}(x^{i}) - y^{i}$)

## Pseudo code of GD
```
GD Function(x, y, initial_w, initial_b, cost, α, num_iterations):
w = w_initial
b = b_initial
J = [] #save the cost function output
P = [] #save w,b parameter values

for i in (num_iterations):
  dj_w, dj_b = gradient #calculate gradient
  w = w - α dj_w
  b = b - α dj_b
  if i < 100000:
    J.append(cost(x,y,w,b))
    P.append([w,b])
return w, b , J, P
```

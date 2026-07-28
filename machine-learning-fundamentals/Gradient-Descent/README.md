# Gradient Descent
As you might remember from previous sections, our defined system learns by adjusting the parameters. In fact, we are looking for the best combination of the parameters together.
How do we know that the model has learned? When it shows the least error or difference between the target value and model prediction. So by minimizing the Cost function, we will 
get more accurate predictions. But we know that this won't happen on its own. Thus, we need an algorithm to update the values for us automatically until it hits the minimum point.

*Gradient Descent* is an algorithm used for this purpose. Perhaps it would be helpful if we reviewed the concept of derivative and gradient before jumping into the algorithm itself.

### Derivative
Derivatives are used to determine the rate of change of a function. For a simple functions such as f(x) = $x^{2}$, by taking the derivative we get f'(x) = 2x which is the rate for which the output of the function changes. As you can see the derivative is a function itself. We can also plot that one.

<img width="370" height="490" alt="f(x) = x^2, f'(x) = 2x graph" src="https://github.com/user-attachments/assets/3b9696ad-1ab2-4d3e-8c74-a70f60b62d7b"/></br>
The pink line is the derivative of f(x). And the other line is the tangent line of the function at (1,1).

# Loss and Cost Function
Inorder to evaluate the predictions of our model, we need to compare the output with the labels
set for each training instance before testing it with unseen data. For this purpose, we use loss
and cost function. The difference between a the prediction and the actual value is called, **prediction error**.
<p align= center>$\hat{y}^{i} - y^{i}$</p></br>
<p align= center>$f_{w,b}(x) - y^{i}$</p>

## Loss function
Loss function measures how wrong the model's answer for a single training instance. In simpler terms,
it compares a single value of the predictions with the labels and returns the difference.
<p align=center>$L^{(i)} = \frac{1}{2} (\hat{y}^{(i)} - y^{(i)})^2$ </p>

## Cost function
A cost function aggregates the losses from all training examples into one value that represents the overall
performance of the model. The purpose of training a model is
to minimize the value of cost function by adjusting the parameters such as weights and bias.
<p align=center>$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2$ </p>

* $\hat{y}^{(i)}$: model's prediction
* $y^{(i)}$: target value (label)
* m: number of traning instances
* J(w,b):the cost fucntion
</br>
⭕ 1/2 is used to simplify the derivative for **gradient escent**.

There are many different cost functions. This one is **MSE**, or ***Mean Squared Error***.

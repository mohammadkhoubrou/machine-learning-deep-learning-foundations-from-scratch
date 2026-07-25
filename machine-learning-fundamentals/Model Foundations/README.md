# Model Foundations
The basic structure of a machine learning model consists of a unit containing a linear equation.
<p align="center">$f_{w,b}(x) = wx + b$</p>


* f : linear function
* x : input
* w : parameter known as **weight**
* b : parameter known as **bias**

The idea behind this system is to feed the algorithm a set of data. Then evaluate the output based on the desired results(labels). During this process and by altering
the value of parameters the equation is set to produce the closest results to the labels. In other words, the system learns to produce the correct answer to the given 
unseen problem or data.

## Inputs and Outputs
Usually the inputs and ouputs of a system are not a single value. Instead, it is a vector of multiple data points. So how do we compute the value of f(x)? Well, you can 
always use a ```for``` loop and it works. If you have 10 data points, no problem. 100? sure. But imagine you have milions of instances! In this case a for loop is not the 
most optional way. Later we will encounter a more efficient way to solve this issue.

## Parameters
This is where the learning happens. By adjusting the parameters we are altering the algorithm to produce the most accurate results possible. *Weights* controls the influence
of the input on the ouput. In simpler terms, it determines how the result of the input should be on the output. *bias* on the other hand, adjusts the ouput regardless of the
input. For instance, even if the input is zero, the ouput can alter it self.

## The relationship between a model and data
The model tries to unreveal the patterns and trends in the data. In the example of predicting the price of a house, the model is trained to predict the price using attributes
such as number of bedrooms, age of the building, etc. By encountering numerous instances with common attributes and the prices assigned to each, the model learns that the more
bedrooms equals higher price. Or older a building gets, the more the price decays.

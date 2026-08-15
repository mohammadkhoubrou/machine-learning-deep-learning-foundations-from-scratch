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
Every neuron in a network is consisted of two back to back sections. After the data is fed to the neuron, it goes through a linear equation. 
<p align= center>
$$
z=\sum_{i=1}^{m}w_{i}x_{j}+b
$$
</p>

- z : result of the linear combination
- $w_i$ : weights
- $x_j$ : training instances
- b : bias


The dot product of the weights and the training examples is then added to a bias.</br>
- Every node(neuron) gets a copy of the training set. Now this set can be the whole dataset which is not common, unless the number of data points is very low. Otherwise, we usually use *batches* to train the data. So we divide the whole data set into multiple batches and in each forward propagation, we'll feed one of the batches as input and update the weights.

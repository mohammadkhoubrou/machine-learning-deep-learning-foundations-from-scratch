# Deep Learning
Deep Learning is the method of creating a network of elements responsible for processing the data, and  making adjustments in the elements based on the result of the output evaluation. In simpler terms, deep learning is a method that teaches a computer certain behaviors.
Deep Learning algorithms can be traced back to 1940s and even before that, there were ideas of such computational methods in 1800s. However, the first trainable neural network structure, close to what we have today, was **the Perceptron** which consists of only one layer of nodes with adjustable weights between the input and output layer.

## What is a **Neural Network**?
In deep learning, the proposed structures include multiple layers where each layer is composed of many nodes. Each node performs certain calculations on the input data and returns the result as an output. Consider a network of processing units; When the input enters each unit, it gets altered and passed on to the next layer as an input. This process repeats untill the data reaches the end of the network. Now in case the output of the network isn's pleasant eanough with what it should've been, the algorithm adjusts the particles of the units and tries again.

## Different types of Learning
We can categorize *Machine learning* and *Deep learning* systems based on the foundation of what they're used for. Mainly three categories define the characteristics of a system; Level of *supervision* in training, *incremental learning* capability, and whether it's *model-based* or *comparison-based*.
### Supervision
ML and DL systems can be examined based on the amount of supervision they got while training. Here we will briefly review a few of those categories such as,
- supervised
- unsupervised
- self-supervised
- reinforcement learning
**We are only looking at a few classes of supervision.**
#### Supervised Learning
In supervised learning, the algorithm requires some more data in addition to the training set. Each set of data has some attributes and these sort of algorithms need correct answer assigned to each instance, which is known as *label*, to compare with their output. So while learning, it compares the solution to the label to adjust and well tune itself. *Classification* and *regression* are classic problems solved by supervised algorithms. A great example of application of this algorithm is predicting the price of the houses in a neighborhood. The dataset is composed of the information of each house(instance) in the neighborhood. Each instance has attributes such as (# of bedrooms, # of floors, age, etc.) and price, which is the label or the correct answer. Now the algorithm learns to get the details of an instance and porpose a price for it based on what it has learnt.

#### Unsupervised Learning
Yes you have guessed right. These algorithms learn with unlabled data. All we need to do is to pass the algorithm our data and it will categorize them based on the similarities and the differences among instances. *Clustering* is an excellent illustration of unsupervised algorithms. *Anamoly detection*, which is often used in spam detection or in cybersecurity to detect suspecious requests to the server is one of the applications of these algorithms. They treat each instance on their own; If it is similar to an already processed instance, it belongs to the same group or cluster. If not, it creates its own cluster.

#### Self-Supervised Learning

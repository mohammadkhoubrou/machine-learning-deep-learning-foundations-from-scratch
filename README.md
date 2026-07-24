# Purpose and Learning Approach
This repository consists of a collection of *systematic* study and *implementation of deep learning and machine learning algorithms* from **scratch**. My goal is to elaborately **document** the concepts I've studied throughout the course of learning Machine learning and Deep learning algorithms, so it serves as a well structured refrence for reviewing the concepts. The repository may also be helpful for those who are learning ML and DL *independently*. My approach is to develop a deeper understanding of the core **mathematical fundamentals** of the concept by studying the mathematical formulas underlying the algorithms and implementing them on my own. Furtheron I practice using established tools and libraries. This repository contains *theoretical description*, *mathematical foundations*, and *from scratch implementations* of the concepts I have studied.

# Deep Learning
Deep Learning is the method of creating a network of elements responsible for processing the data, and  making adjustments in the elements based on the result of the output evaluation. In simpler terms, deep learning is a method that teaches a computer certain behaviors.
Deep Learning algorithms can be traced back to 1940s and even before that, there were ideas of such computational methods in 1800s. However, the first trainable neural network structure, close to what we have today, was **the Perceptron** which consists of only one layer of nodes with adjustable weights between the input and output layer.

## What is a **Neural Network**?
In deep learning, the proposed structures include multiple layers where each layer is composed of many nodes. Each node performs certain calculations on the input data and returns the result as an output. Consider a network of processing units; When the input enters each unit, it gets altered and passed on to the next layer as an input. This process repeats until the data reaches the end of the network. Now in case the output of the network isn's pleasant enough with what it should've been, the algorithm adjusts the particles of the units and tries again.

## Different types of Learning
We can categorize *Machine learning* and *Deep learning* systems based on the foundation of what they're used for. Mainly three categories define the characteristics of a system; Level of *supervision* in training, *incremental learning* capability, and whether it's *model-based* or *comparison-based*.
### Supervision
ML and DL systems can be examined based on the amount of supervision they got while training. Here we will briefly review a few of those categories such as,
- supervised
- unsupervised
- self-supervised
- reinforcement learning</br>
**We are only looking at a few classes of supervision.**
  
#### Supervised Learning
In supervised learning, the algorithm requires some more data in addition to the training set. Each set of data has some attributes and these sort of algorithms need correct answer assigned to each instance, which is known as *label*, to compare with their output. So while learning, it compares the solution to the label to adjust and well tune itself. *Classification* and *regression* are classic problems solved by supervised algorithms. A great example of application of this algorithm is predicting the price of the houses in a neighborhood. The dataset is composed of the information of each house(instance) in the neighborhood. Each instance has attributes such as (# of bedrooms, # of floors, age, etc.) and price, which is the label or the correct answer. Now the algorithm learns to get the details of an instance and propose a price for it based on what it has learnt.

#### Unsupervised Learning
Yes you have guessed right. These algorithms learn with unlabeled data. All we need to do is to pass the algorithm our data and it will categorize them based on the similarities and the differences among instances. *Clustering* is an excellent illustration of unsupervised algorithms. *anomaly detection*, which is often used in spam detection or in cybersecurity to detect suspicious requests to the server is one of the applications of these algorithms. They treat each instance on their own; If it is similar to an already processed instance, it belongs to the same group or cluster. If not, it creates its own cluster.

#### Self-Supervised Learning
Self-supervised learning is an approach in which the model is trained to learn from unlabeled data by creating its own representation (feature represantation: attributes represented as vectors of numbers after learning). This looks similar to *unsupervised learning*. Yes it does, however, the difference is that there are no labels in unsupervised learning, but in this approach the model learns to create labels out of the data itself. Then, the model can be fine-tuned to be used in other tasks such as detection, classification, or segmentation. A famous practice of this method is predicting a covered part of an image. Consider an image of a plant with a giant black square in the middle of it. By training a model on a set of instances the model learns to predict the covered part of the image. Throughout this process, the model has learnt the features of the instances and it is now pretty familiar with its attributes. Therefore, it is used for classifying or detecting specific objects in images such as plant disease detection.

#### Reinforcement Learning
In this approach the system, better known as *the agent*, is put in an environment where it has to observe, choose and action and perform. How does it learn then? There is supposed to be a *reward* of a *penalty* for each action. When the agent makes the right desicion, it gets a reward and it punished for each mistake. Thus, it develops a *policy*. A policy is the set of actions that an agent can select for a given situation. For instance, a robot is trying to navigate to the destination. There are a certain number of states(e.g. current position, nearby obstacle, destination), a set of actions(e.g. move forward, turn left, turn right, stop), and rewards. I suggest you to search reinforcement learning for game agents on YouTube for a fun experience and better understanding the concept.

## Incremental Learning Capability (Batch Vs Online)
In this section we classify systems into 2 groups in terms of their ability to learn from a stream of incoming data.

### Batch Learning
This method is also known as *Ofline Learning*. The system is incapable of learning continuesly. The model must be trained on the whole dataset and then the stops and the model is deployed. While the model is working, learning is stoped. In this case the model tends to underperform overtime, simply because it has stopped learning but the world continues. This problem is known as *model rot* or *data drift*. The solution to this problem is constantly training new models on new data to preserve the performance quality. A clear example of this is the models designed for financial markets. Such a system makes predictions based on the previous records, therefore, it should be trained with updated data.
### Online Learning
In this approach, the model is fed either small groups of samples called *mini-batches* or individual data instances. The learning is continues, cheaper, and faster. It is usually used when the system should be adapted to new data like financial market prediction tasks. Online learning is memory efficient, because the input batches are added and trained on once at a time. As you might have already guessed, such systems are very sensitive to input data quality. If the data is bad, the system quality will decline and since it is live, the users can easily notice the decay.

## Instance-Based vs Model-Based Learning
One way to measure the quality of a system is to evaluate the way it generalizes on the unseen before instances. When a model(usually prediction models) performs well on the new data after training and makes accurate predictions, it is said that it has generalized. There are two ways of generalization:

### Instance-Based Learning
In this method, the model tries comparing new instances with the flagged ones. Then, checks for similarities and if they match, the new instance belongs to that cluster or group. Consider a spam detector. It takes an labeled instance and not only looks for exact matches of that email but also the ones that are similar using a similarity measure.

### Model-Based Learning
Using a model-based learning system, we define the type of algorithm best fit to the problem in hand. Then, after cleaning and preparing the data, we start the training. The model finds a trend in the data, then based on the input, generates the most probable output. If the data quality is good and the model is trained well, the output will be closer to the desired result.
## Refrences
- Andrew NG Machine Learning and Deep Learning Courses
- Hands-On Machine Learning with Scikit-Learn, Kera & TensorFlow by Aurélien Géron

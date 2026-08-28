# Convolutional Neural Network (CNN)
*Convolutional Neural Networks* are a type of artificial intelligence algorithms designed to process and understand the visual and signal data. CNNs learn by passing a filter better known as the *kernel* over the different sections of the image or a signal. In this method, the system usually interacts with the different parts of the input data more than once. An image data has two spatial dimensions, $\mathbb{R}^{HxW}$. For example, $X \in \mathbb{R}^{28x28}$. Filters will detect 2-D patterns such as edges, corners, textures, shapes, and complex visual structures.
Whereas, a signal has one primary dimension, $$X = [x_{1}, x_{2}, x_{3}, \dots, x_{N}]$$. For example an N HZ EEG signal, the dimensions of our data is $X \in \mathbb{R}^{1xN}$, meaning we have sampled N points per second. EEG however, isn't a 1-D signal. It includes different channels. So we can either process one channel at a time, or stach them up and extract the attributes simultaneously. 

An image can be represented as a matrix. For instance a gray scale image is represented bellow, where each element represents a pixel. For pixel values, ``` black = 0 ``` and ``` white = 1 ``` and any number between those determines the brightness of the pixel.

$$X =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

For RGB images, we have 3 channels.

$$X \in \mathbb{R}^{HxWx3}$$
$$X_{i,j} = [R_{i,j}, G_{i,j}, B_{i,j}]$$

## Kernel
Kernel is the filter that we move on the data to do the necessary operations. For each concatenating window, the algorithm first operates a *Hadamard* (element-wise) multiplication of the corresponding elements, and then, returns the sum of the elements of the created window. For Horizontal edges the following filter is most popular,

$$K =
\begin{bmatrix}
1 & 1 & 1 \\
0 & 0 & 0 \\
-1 & -1 & -1
\end{bmatrix}
$$
<p align= center>Top row is the bright side, bottom is the dark side.</p>

For Vertical edges,

$$K =
\begin{bmatrix}
1 & 0 & -1 \\
2 & 0 & -2 \\
1 & 0 & -1
\end{bmatrix}
$$
<p align= center>Sobel filter</p>

$$K =
\begin{bmatrix}
3 & 0 & -3 \\
10 & 0 & -10 \\
3 & 0 & -3
\end{bmatrix}
$$
<p align=center>Scharr filter</p>

We can treat the elements in the kernel as parameters. So with proper learning techniques, the system learns what to extract at every layer of the network.

$$K =
\begin{bmatrix}
w_1 & w_2 & w_3 \\
w_4 & w_5 & w_6 \\
w_7 & w_8 & w_9
\end{bmatrix}
$$

## Padding
The dimensions of the ouput of a layer is determined as follows,
```nxn * fxf = n-f+1```

- n : size of the input image
- f : size of the filter

Two problems emerge during convolution. First, *Shrinking image*. Every time we convolve the image, it shrinks and gets smaller. Second, *information loss* Since the corner pixels are only used once, we're practically throwing away a lot of information. Padding fixes these issues. 

Padding is to add one or more pixel rows to the edges of the image. By convention we pad with zeros and the size of the output is,
```n+2p x n+2p```
### Valid and Same Convolution
valid convolution doesn't use padding and we saw how to compute the output size above.
*Same* convolution is when we want to get an output the same size as the input. So we must determine the value of padding,
```
n + 2p -f + 1 = n

p = (f-1)/2
```

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



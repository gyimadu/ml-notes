**LINEAR REGRESSION**
A linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant term called the bias term (also called the intercept term).
$$
\hat{y} = \theta_0 + \theta_2x_2 + ... + \theta_nx_n
$$
$\hat{y}$  is the predicted value
n is the number of features
$x_i$ is the $i^{th}$ feature value
$\theta_j$  is the $j^{th}$ model parameter (including the bias term $\theta_0$ and the feature weights $\theta_1$, $\theta_2$, ..., $\theta_n$)

*Linear Regression model prediction (vectorized form)*
$$
\hat{y} = h_\theta(x) = \theta^T + x
$$

$\theta$ is the model's parameter vector, containing the bias term $\theta_0$ and the feature weights $\theta_1$ to $\theta_n$ 
$\theta^T$ is the transpose of $\theta$ (a row vector instead of a column vector).
**x** is the instance's feature vector, containing $x_0$ to $x_n$, with $x_0$ always equal to 1
$\theta^T$ $\cdot$ **x** is the dot product of $\theta^T$ and **x**

To train a linear regression omdel, you need to find the value of $\theta$ that minimizes the Root Mean Square Error (RMSE). In practice, it is simper to minimize the Mean Square Error (MSE) than the RMSE, and it leads to the same result.

*MSE cost function for a Linear Regression model*
$$
MSE(X, h_\theta) = \frac{1}{m}\sum_{i=1}^m (\theta^T\cdot x^{(i)} - y^{(i)})^2
$$

**The Normal Equation**
Closed form solution for finding the value of $\theta$ that minimizes the cost functions
$$
\hat{\theta} = (X^T\cdot X)^{-1}\cdot X^T\cdot y
$$
$\hat{\theta}$  is the value of $\theta$ that minimizes the cost function.
**y** is the vector of target values containing $y^{(1)}$ to $y^{(m)}$

The normal equations is linear with regards to the number of instances in teh training set (it is $O(m)$), so it handles large training sets efficiently, provided they can fit into memory.

**GRADIENT DESCENT**

**Batch Gradient Descent (BGD)**
Batch gradient descent uses the whole batch of the training set at every step which makes it terribly slow on very large training sets. It is however better than using the Normal Equation.

To implement gradient descent, you need to calculate how much the cost functions will change if you change $\theta_j$ just a little bit.

*Partial derivative of the cost function*
$$
\frac{\partial}{\partial\theta_j}MSE(\theta) = \frac{2}{m}\sum_{i-1}^m (\theta^T \cdot x^{(i)} - y^{(i)})x_j^{(i)}
$$

*Gradient vector of the cost function*
$$
\nabla_\theta MSE(\theta) = \begin{pmatrix}\frac{\partial}{\partial\theta_0}MSE(\theta)\\ \frac{\partial}{\partial\theta_1}MSE(\theta) \\ .\\.\\. \\\frac{\partial}{\partial\theta_n}MSE(\theta)  \end{pmatrix} = \frac{2}{m}X^T\cdot (X\cdot \theta - y)
$$


*Gradient Descent Step*
$$
\theta^{(next\:step)} = \theta - \eta\nabla_\theta MSE(\theta)
$$

$\eta$ is the learning rate, which helps determine the size of the downhill step

*Convergence Rate*
When the cost functions if convex, and its slope doesn't change abruptly, then it can be shown that Batch Gradient Decent with a fixed learning rate has a convergence rate of $O(\frac{1}{iterations})$ .

**Stochastic Gradient Descent (SGD)** 
Stochastic Gradient Descent, unlike BGD, picks a random instance in the training set at every step and computes the gradients based only on that single instance. 

This makes it possible to train on huge training sets, since only one instance needs to be in memory at each iteration (SGD can be implemented as an out-of-core algorithm). 

It's however much less regular than BGD: instead of gently decreasing until it reaches the minimum, the cost functions will bounce up and down, decreasing only on average. 

It ends up very close to the minimum overtime, but once it gets there it will continue to bounce around, never settling down. Final parameters are good, but not optimal. This can be solved using a process called $simulated\:annealing$.

**Mini-batch Gradient Descent (MIni-batch GD)**
Mini-batch GD computes the gradients on small random sets of instances called mini-batches. 

The main advantage of Mini-batch over Stochastic GD is that you can get a performance boost from hardware optimization of matrix operations, especially when using GPU's.

Mini-batch is less erratic than Stochastic GD and therefore ends up walking a bit closer to the minimum than SGD. On the other hand, it may be harder for it to escape from local minima.

**POLYNOMIAL REGRESSION**


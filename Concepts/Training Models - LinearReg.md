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

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta=0.1)
sgd_reg.fit(X, y_ravel())
```

**Mini-batch Gradient Descent (MIni-batch GD)**
Mini-batch GD computes the gradients on small random sets of instances called mini-batches. 

The main advantage of Mini-batch over Stochastic GD is that you can get a performance boost from hardware optimization of matrix operations, especially when using GPU's.

Mini-batch is less erratic than Stochastic GD and therefore ends up walking a bit closer to the minimum than SGD. On the other hand, it may be harder for it to escape from local minima.

**POLYNOMIAL REGRESSION**

Polynomial Regression is a way to use a linear model to fit nonlinear data by adding powers of each features, then training the linear model on this extended set of features.

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# X_poly now contains the original feature X plus the square of this feature. Now you can fit a LinearRegression model to this extended training data

lin_reg = LineaRegression()
lin_reg.fit(X_poly, y)
```

*The Bias/ Variance Tradeoff*

A model's generalization error can be expressed as the sum of three very different errors.
**Bias:**
This  is due to wrong assumptions such as assuming the data is linear when it is actually quadratic. A high-bias model is most likely to under-fit the training data.

**Variance:**
This is due to the model's excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance, and thus overfit the training data.

**Irreducible Error:**
This is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g fix the data sources, such as broken sensors, or detect and remove outliers)

**REDULARIZED LINEAR MODELS**
For a linear model, regularization is typically achieved by constraining the weights of the model.

**Ridge Regression:**
Also called $Tikhonov\:regularization$. Ridge regression is a regularized version of Linear Regression: a regularization term equal to $\alpha\sum_{i=1}^n \theta_i^2$  is added to the cost function. This forces the algorithm to not only fit the data but also keep the model weights as small as possible. 

The regularization term should only be added to the cost function during training.

The hyper-parameter $\alpha$ controls how much you want to regularize the model. If $\alpha = 0$ then Ridge Regression is just Linear Regression. If $\alpha$ is very large, then all weights end up very close to zero and the result is a flat line going through the data's mean. 

*Ridge Regression cost function*
$$
J(\theta) = MSE(\theta) + \alpha\frac{1}{2}\sum_{i=1}^n\theta_i^2
$$

It is important to scale the data (e.g using a StandardScaler) before performing Ridge Regression, as it is sensitive to the scale of the input features.

*Ridge Regression closed-form solution*
$$
\hat\theta = (X^T\cdot X + \alpha A)^{-1}\cdot X^T\cdot y 
$$
```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[some value]])
```
*and using SGD*
```python
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[some value]])
```

**Lasso Regression:**
$Least\:Absolute\:Shrinkage\:and\:Selection\:Operator\:Regression$ (simply Lasso Regression) is another regularized version of Linear Regression. It adds a regularization term to the cost function, but it uses the $l_1$ norm of the weight vector instead of half the square of the $l_2$ norm.

*Lasso Regression cost function*
$$
J(\theta) = MSE(\theta) + \alpha\sum_{i=1}^n \vert{\theta_i}\vert
$$
An important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features (i.e set them to zero). In other words, Lasso Regression automatically performs feature selection and outputs a $sparse\:model$ (i.e. with few nonzero feature weights).

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[some value]])
```

**Elastic Net:**
Elastic Net is the middle ground between Ridge Regression and Lasso Regression. The regularization term is a simple mix of bothRidge and Lasso's regularization terms, and you can control the mix ratio $r$. When $r = 0$, Elastic Net is equivalent to Ridge Regression, and when $r = 1$, it is equivalent to Lasso Regression.

*Elastic Net cost function*
$$
J(\theta) = MSE(\theta) + r\alpha\sum_{i=1}^n \vert{\theta_i}\vert + \frac{1 - r}{2}\alpha\sum_{i=1}^n\theta_i^2
$$

A little bit of regularization is always preferable, so avoid using plain Linear Regression. Ridge is a good default, but if it's suspected that only a few features are actually useful, you should prefer Lasso or Elastic Net since they tend to reduce the useless features' weights down to zero. 

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[some value]])
```

In general, Elastic Net is preferred over Lasso since Lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated.

**Early Stopping:**
This is a way to regularize iterative learning algorithms such as Gradient Descent by stopping training as soon as the validation error reaches a minimum. 

*Basic implementation of early stopping*

```python
from sklearn.base import clone

sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None,
					  learning_rate="constant", eta=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
	sgd_reg.fit(X_train_poly_scaled, y_train)
	y_val_predict = sgd_reg.predict(X_val_poly_scaled)
	val_error = mean_squared_error(y_val_predict, y_val)
	if val_error < minimum_val_error:
		minimum_val_error = val_error
		best_epoch = epoch
		best_model = clone(sgd_reg)

```
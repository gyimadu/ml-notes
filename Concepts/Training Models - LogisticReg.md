Also called Logit Regression; used to estimate the probability that an instance belongs to a particular class. If the estimated probability is greater than 50%, then the model predicts that the instance belongs to that class (positive class, labeled "1"), or else it predicts that it does not (negative class, labelled "0"). This makes it a binary classifier.


TRAINING MODELS EXERCISES AND SOLUTIONS
1. What Linear Regression training algorithm can you use if you have a training set with millions of features?

If you have a training set with millions of features you can use Stochastic Gradient Descent or Mini-Batch Gradient Descent, and perhaps Batch Gradient Descent if the training set fits into memory. But you cannot use the Normal Equation because the computational complexity grows quickly (more than quadratically) with the number of features.

Calculating the $(X^TX)^{-1})$, the inverse of the matrix has a computational complexity typically of about $O(n^{2.4})$ to $O(n^3)$. If you have millions of features($n = 10^6$), this step becomes extremely expensive, as the number of operations scales cubically with $n$. Additionally, constructing and storing $X^TX$ in memory is infeasible for such large $n$, because it would require trillions of storage units. Therefore the Normal Equation becomes computationally impractical when the number of features is large.

Gradient Descent methods don't require directly computing a matrix inverse. Instead, they iteratively optimize the cost function by following the gradient of the error.

2. Suppose the features in your training set have very different scales. What algorithms might suffer from this, and how? What can you do about it?

Gradient descent algorithms converges more slowly because features with larger scales dominate the gradient, causing an uneven optimization landscape.

If the features in the training set have very different scales, the cost function will have the shape of an elongated bowl, so the Gradient Descent algorithms will take a long time to converge. To solve this you should scale the data before training the model.

3. Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?

Gradient Descent cannot get stuck in a local minimum when training a Logistic Regression model because the cost function is convex.

4.  Do all Gradient Descent algorithms lead to the same model provided you let them run long enough?

*If the optimization problem is convex (such as Linear Regression or Logistic Regression), and assuming the learning rate is not too high, then all Gradient Descent algorithms will approach the global optimum and end up producing fairly similar models. However, unless you gradually reduce the learning rate, Stochastic GD and Mini-batch GD will never truly converge; instead, they will keep jumping back and forth around the global optimum. This means that even if you let them run for a very long time, these Gradient Descent algorithms will produce slightly different models.*

5. Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?

I*f the validation error consistently goes up every epoch, then one possibility is that the learning rate is too high and the algorithm is diverging. If the training error also goes up, then this is clearly the problem and you should reduce the learning rate. However, if the training error is not going up, then your model is overfitting the training set and you should stop training.*

6.  Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?

*Due to their random nature, neither Stochastic GD nor Mini-batch GD is guaranteed to make progress at every single training iteration. So if you immediately stop training when the validation error goes up, you may stop much too early, before the optimum is reached. A better option is to save the model at regular intervals, and when it has not improved for a long time, you can revert to the best saved model.*

7. Which Gradient Descent algorithm will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make sure the others converge as well?

*Stochastic GD has the fastest training iteration since it considers only one training instance at a time, so it is generally the first to reach the vicinity of the global optimum. However, only Batch GD will actually converge, given enough training time. As mentioned, Stochastic GD and Mini-batch GD will bounce around the optimum, unless you gradually reduce the learning rate.*

8. Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?

*If the validation error is much higher than the training error, it is highly likely that the model is overfitting the training set. One way to try to fix this is to reduce the polynomial degree: a model with fewer degrees of freedom is less likely to overfit. Another thing to try is to regularize the model - for example, by adding an $l2$ penalty (Ridge) or an $l1$ penalty (Lasso) to the cost function. Lastly, the training set can be reduced.*

9. Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization parameter $\alpha$ or reduce it?

If both the training error and the validation error are almost equal and fairly high, the model is likely under-fitting the training set, which means it has a high bias. You should try reducing the regularization hyper-parameter $\alpha$.

1. Why would you want to use:
		- Ridge Regression instead of Linear Regression?
		A model with some regularization typically performs better than a model without any regularization, so you should generally prefer Ridge Regression over plain Linear Regression.
		- Lasso instead of Ridge Regression?
		Lasso Regression uses an $l1$ penalty, which tends to push the weights down to exactly zero. This leads to sparse models, where all weights are zero except for the most important weights. This is a way to perform feature selection automatically, which is good if you suspect that only a few features actually matter.
		- Elastic Net instead of Lasso?
		Lasso may behave erratically in some cases (when seeral features are strongly correlated or when there are more features than training instances). However, it does add an extra hyper-parameter to tune. If you just want Lasso without the erratic behavior, you can just use Elastic Net with an $l1$ ratio close to 1.
1. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or one Softmax Regression classifier?
2. Implement Batch Gradient descent with early stopping for Softmax Regression (without using scikit-learn).
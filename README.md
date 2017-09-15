# numpy-linear-regression

This repository uses linear regression implemented using Numpy to demonstarate gradient descent. Gradient descent will be used as our optimization startegy for linear regression. We will be using linear regression to draw the line of best fit to measure the relationship between student test scores and the amount of hours studied.

## Methodology

Like all machine learning algorithms we first need to get our data. The data file, `data.csv` contains student test scores and the number of hours  studied. We have 100 rows of data. Let's start by defining our main function and import everything from numpy

```python
from numpy import *

if __name__ == '__main__':
    run()
```

The `run` function will contains our initial values, our hyperparameters and function calls. Lets define our `run` function.

```python
def run():
    points = genfromtxt('data.csv', delimiter=',')
    # Hyperparameter
    learning_rate = 0.0001
    # Initial values: y = mx + c
    initial_c = 0
    initial_m = 0
    # Iterations
    num_iterations = 1000
    # Optimal values for m and c
    [c, m] = gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iterations)
    # Results
    error = compute_error_for_points(c, m, points)
    print("Optimized after {0} iterations: m = {1}, c = {2} and error = {3}".format(num_iterations, m, c, error))
```

As stated earlier, like all ML algorithms we first need to get our data in a format that we can handle, so we use `genfromtext` to load data from the `data.csv` file and since the data is in a `csv` format, we use the comma as a `delimiter`. Then we define a hyperparameter, our `learning_rate` determines how fast our model is able to converge (find the optimal value of m and c). The we define the initail values of `m` and `c` and we initialize the line as a horizontal line. The `num_iterations` determines the number of times we run our optimizer, since this is a small data set we should be able to reach convergence with just a 1000 iterations. The `gradient_descent_runner` runs the gradient descent optimizer for the number of iterations specified, in our case for a 1000 iterations. Then we compute the error and print out the final, optimized values. Now we need to define the functions that we are calling.

```python
def gradient_descent_runner(points, starting_c, starting_m, learning_rate, num_iterations):
    c = starting_c
    m = starting_m
    # Iterate
    for i in range(num_iterations):
        c, m = step_gradient(c, m, array(points), learning_rate)
    return [c, m]
```

Before we proceed onto defining the `step_gradient` function which is our optimizer, we need understand the process of gradient desecent. Since we are starting off with a horizontal line, we need a way to adjust the `m` and `c` parameters so that it will produce the line of best fit for our data. To do this, we first need to know how far off the mark our line is from the data points at a specific `x` value and we compute this error value using what's known as the sum of squared errors. The equation for the sum of squred errors as as follows:

<div align="center">
    <br><img src="https://cldup.com/CL6TX3cVvZ.png" width="401.7" height="72.9"><br><br>
</div>

Essentially, we subtract the `y` value of our line at a specific `x` value from the `y` value of data point to obtain the margin of error. The `y` value of our line is defined as `mx + c`. Then we square the error because we only want positive values when we sum and since we are only interested in the magnitude of the value and not the value itself. Then we sum the error across all points and divide them by the number of points to obtain the total error. Now that we have calculated the total error, our goal would be to minimize the error value. Before we look at how we can minimize the error value, let's look at this excellent visualization [An Introduction To Gradient Descent And Linear Regression,  https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/].

<div align="center">
    <br><img src="https://spin.atomicobject.com/wp-content/uploads/gradient_descent_error_surface.png"><br><br>
</div>

The visualization shows all the possible values of `m`, `c` (shown as `b` in the visualization) and `error`. We know that our goal is to adjust the `m` and `c` values to obtain lowest possible error value. The lowest possible error value is at the bottom of the curve, at the dark blue region of the graph. This region is know as the local minima. The bottom of the curve has a gradient of 0 and if we were to find the gradient of the curve at specific `m` and `c` values and adjust the `m` and `c` values such that the gradient eventually reaches 0, we will be able to achieve convergence easily. To do this, we need to find the gradient of `m` and `c`. The gradients of `m` and `c` are partial derivatives and they can be computed using the follwing equations.

<div align="center">
    <br><img src="https://cldup.com/Pv9bByAJvW.png" width="327" height="72.9"><br>
</div>

<div align="center">
    <br><img src="https://cldup.com/xb0xd9mbNU.png" width="360.6" height="72.9"><br><br>
</div>



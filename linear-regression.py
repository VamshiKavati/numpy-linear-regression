from numpy import *


def compute_error_for_points(c, m, points):
    totalError = 0
    # Iterate
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + c)) ** 2
    return totalError / float(len(points))


def step_gradient(c_current, m_current, points, learning_rate):
    # Gradient descent
    c_gradient = 0
    m_gradient = 0
    N = float(len(points))
    # Iterate
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        c_gradient += -(2 / N) * (y - ((m_current * x) + c_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + c_current))
    # Update m and c
    new_c = c_current - (learning_rate * c_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_c, new_m]


def gradient_descent_runner(points, starting_c, starting_m, learning_rate, num_iterations):
    c = starting_c
    m = starting_m
    # Iterate
    for i in range(num_iterations):
        c, m = step_gradient(c, m, array(points), learning_rate)
    return [c, m]


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
    [c, m] = gradient_descent_runner(
        points, initial_c, initial_m, learning_rate, num_iterations)
    # Results
    error = compute_error_for_points(c, m, points)
    print("Optimized after {0} iterations: m = {1}, c = {2} and error = {3}".format(num_iterations, m, c, error))


if __name__ == '__main__':
    run()

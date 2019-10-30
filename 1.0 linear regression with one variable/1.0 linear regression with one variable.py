import numpy as np
from matplotlib import pyplot as plt

# data process
data = np.loadtxt('../data_sets/ex1data1.txt', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]
x_plot = x_data  # for plotting the hypothesis function
print('x_data.shape:', x_data.shape)
print('y_data.shape:', y_data.shape)
m = y_data.size

# plot data
fig = plt.figure(0, figsize=(8, 8))
ax = fig.add_subplot(2, 1, 1)
plt.xlabel('population')
plt.ylabel('profit')
ax.scatter(x_data, y_data, color = 'b', label = 'Training data')
plt.ion()  # transform the block mode to interactive mode

# add one dimensional onto x_data with x0 = 1
x0 = np.ones(x_data.size)
print('x0.shape:', x0.shape)
x_data = np.c_[x0, x_data]
print('x_data.shape:', x_data.shape)

# initialization
theta = np.zeros(2)
print('theta.shape:', theta.shape)
num_iteration = 1000
alpha = 0.02
J = np.zeros(num_iteration)


def cost_function(theta, x = x_data, y = y_data, m = m):
    # hypothesis
    h_x = x@theta  # x->[97,2] theta->[2,1] h_x->[97,1]
    # evaluate the cost function
    return np.sum((h_x - y)**2)/(2*m)


def gradient_descent(theta, alpha, num_iteration):
    for i in range(num_iteration):
        J[i] = cost_function(theta)
        h_x = x_data@theta
        theta = theta - (alpha/m) * (x_data.T @ (h_x - y_data))
        print('num_iteration:{}, cost:{} '.format(1+i, J[i]))
        # plot the hypothesis
        if i % 10 == 0:
            try:
                # erase last lines
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot(x_plot, h_x, color = 'r', label = 'Hypothesis', lw = 5)
            plt.pause(0.1)
            plt.legend(loc=2)
    return theta


# build the model
print('build the model ... ')
theta = gradient_descent(theta, alpha, num_iteration)
# plot the cost function line
ax1 = fig.add_subplot(2, 1, 2)
plt.xlabel('num_iteration')
plt.ylabel('cost values')
ax1.plot(np.arange(num_iteration), J, lw = 3)
print('Final hypothesis function:h(x) = {}x0 + {}x1'.format(theta[0], theta[1]))
print(' train successfully !! ')
plt.ioff()
plt.show()


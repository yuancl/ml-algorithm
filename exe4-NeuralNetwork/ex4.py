# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

from scipy.io import loadmat

from nn_cost_fun import nn_cost_fun
from rand_initialize_weights import rand_initialize_weights
from display_data import display_data
from predict import predict
from compute_numerical_grad import compute_numerical_grad

input_layer_size = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10

data = loadmat('data/ex4data1.mat')
X = data['X']
y = data['y']%10
# y = data['y']
# print 'y = ',np.unique(y)
print 'X.shape = ',X.shape
print 'y.shape = ',y.shape

weights = loadmat('data/ex4weights.mat')
weights.keys()
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
print 'Theta1.shape = ',Theta1.shape
print 'Theta2.shape = ',Theta2.shape

init_params = np.r_[Theta1.ravel(),Theta2.ravel()]
print nn_cost_fun(init_params, 400, 25, 10, X, y, 0)[0]

# """
# Initializing Parameters Rand
rand_theta1 = rand_initialize_weights(400, 25)
rand_theta2 = rand_initialize_weights(25, 10)
rand_params = np.r_[rand_theta1.ravel(), rand_theta2.ravel()]
# print nn_cost_fun(rand_params, 400, 25, 10, X, y, 0)[0]




# ================= compute_numerical_gradient... =================
# If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9) Relative Difference
sample = np.random.choice(X.shape[0], 10)
XX = X[sample]
yy = y[sample]
nn_param_grad = nn_cost_fun(rand_params, 400, 25, 10, XX, yy, 0)[1]
number_param_grad = compute_numerical_grad(rand_params, 400, 25, 10, XX, yy, 0)
diff = np.abs(number_param_grad-nn_param_grad)/(np.abs(number_param_grad)+np.abs(nn_param_grad))
print 'number_param_grad diff is: ',diff[0:100]



print 'Training Neural Network...'
l = 1
result = opt.minimize(fun=nn_cost_fun, x0=rand_params,
                      args=(input_layer_size, hidden_layer_size, num_labels, X, y, l),
                      method='TNC', jac=True, options={'maxiter': 150})
params_trained = result.x

Theta_1_trained = params_trained[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
Theta_2_trained = params_trained[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

print 'Visualizing Neural Network...'
# plt.figure()
# display_data(Theta_1_trained[:, 1:], padding=1)
# plt.show()


# =================Implement Predict =================
pred = predict(Theta_1_trained, Theta_2_trained, X)
# print 'Training Set Accuracy:', np.mean(pred == y) * 100
print 'accuracy is {}%'.format(np.mean(pred == y.ravel())*100)
# """

print 'pred front 10 is : ', pred[3800:4050]
print '   y front 10 is : ', y.ravel()[3800:4050]

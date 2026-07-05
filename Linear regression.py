import numpy as np

y = np.array([300,450,600])   #True values
y = y.reshape(3,1)
# print(y)

X_flat = np.array([1000,1500,2000])  #The parameter (size of house in sq. ft)
X_col = X_flat.reshape(3,1)
# print(X_col)

ones = np.ones((3,1))
# print(ones)

'''Making the column of features'''
X = np.c_[ones, X_col]  # The entire parameter block
np.set_printoptions(suppress=True)
# print(X,X.shape)

'''Initializing the weights(theta) to 0'''
theta = np.zeros((2,1))
# print(theta)

# '''Calculate the predictions'''
# predictions = X.dot(theta)
# # print(predictions)

# error = predictions-y
# # print(error, error.shape)

# '''Gradients calculation'''
# gradients = X.T.dot(error)*(1/m)
# # print(gradients)

m = len(y)
'''Gradient descent loop'''
iterations = 10
alpha = 0.000000001
for i in range(iterations):
    # make predictions
    predictions = X.dot(theta)

    # find the error
    error = predictions-y
    np.set_printoptions(suppress=True)

    # calculate the gradients
    gradients=(1/m)*(X.T.dot(error)) 
    np.set_printoptions(suppress=True)

    # Update the weights
    theta = theta - alpha*(gradients)
    np.set_printoptions(suppress=True)
    print(f"For {i+1}th epoch the weights are {theta}")

print(f"After {i+1} iterations the weights are {theta}")
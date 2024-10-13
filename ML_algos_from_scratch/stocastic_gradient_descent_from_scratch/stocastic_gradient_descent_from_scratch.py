'''
This program performs SGD on a custom dataset that is produced using the random function. 
It can be extrapolated to any type of dataset in a Regression model
'''

import tensorflow as tf
import random
import numpy as np


#Create the random dataset to test the model on the regression task

#This is a very abstract dataset and is just used for testing the model
x_train = np.random.uniform(0, 10, size=(100, 1)).astype(np.float32)
y_train=x_train+1
#Define the model
model=tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu',input_shape=(1,)),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1)
    ])

#Compile the model with the SGD opimizer and the MSE loss function
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')

#Fitting the model to the training data
model.fit(x_train,y_train, epochs=100)

# Evaluate the model
loss = model.evaluate(x_train, y_train)
print(f"Loss: {loss}")


# Make predictions
predictions = model.predict(x_train)
print("Predictions:", predictions)


#Explanation of what SGD does in comparison to traditional Gradient Descent:
'''
Gradient Descent is an algorithm that is used to minimize the loss function to increase
the accuracy of the predictions by the model. This is done by modifiying the weights of the model
SGD and GD differ in how they update these parameters during each iteration. 

GD computes the gradient of the descent by evaluating the entire dataset at once and there is
no recursion or iteration

This is not at all optimal as we are not aware of the specifics of the dataset. 
And also is computationally inefficient as it requires evaluating the entire dataset at every iteration

But SGD does the same thing iteratively or in other words for each and every data point.
Updates the params using only one training example at a time. SGD introduces noise into the training data
which can help it to avoid shallow minimas and look for better, more optimal solutions. 

But both of them have their own pros and cons.

There is a third type of Gradient Descent called Mini-Batch Gradient Descent 
which is a combination of both GD and SGD and takes a fixed 'k' data points at a time
and calculates the loss function iteratively for (n/k) times(n data points in the entire dataset)

'''
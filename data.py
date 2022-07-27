import numpy as np 
import math
import random

#fucntion for joint pdf of bivariate normal distribution with standard deviation of 1 for both x and y 
def f(x, y, mu_x, mu_y, sigma_x=1, sigma_y=1):
    pi = math.pi
    return (1/(2*pi*sigma_x*sigma_y))*math.exp((-1/2)*(((x-mu_x)/sigma_x)**2+((y-mu_y)/sigma_y)**2))

#creates an array of values whose position in x-y corrdinates (laid out in write-up) 
#corresponds to the x and y from pdf of bivarite distribution with given means 
def createBivariateArray(mu_x, mu_y):
    x=[ 
        f(-4,4,mu_x,mu_y), f(-3,4,mu_x,mu_y), f(-2,4,mu_x,mu_y), f(-1,4,mu_x,mu_y), f(0,4,mu_x,mu_y), f(1,4,mu_x,mu_y), f(2,4,mu_x,mu_y), f(3,4,mu_x,mu_y), f(4,4,mu_x,mu_y), 
        f(-4,3,mu_x,mu_y), f(-3,3,mu_x,mu_y), f(-2,3,mu_x,mu_y), f(-1,3,mu_x,mu_y), f(0,3,mu_x,mu_y), f(1,3,mu_x,mu_y), f(2,3,mu_x,mu_y), f(3,3,mu_x,mu_y), f(4,3,mu_x,mu_y),
        f(-4,2,mu_x,mu_y), f(-3,2,mu_x,mu_y), f(-2,2,mu_x,mu_y), f(-1,2,mu_x,mu_y), f(0,2,mu_x,mu_y), f(1,2,mu_x,mu_y), f(2,2,mu_x,mu_y), f(3,2,mu_x,mu_y), f(4,2,mu_x,mu_y),
        f(-4,1,mu_x,mu_y), f(-3,1,mu_x,mu_y), f(-2,1,mu_x,mu_y), f(-1,1,mu_x,mu_y), f(0,1,mu_x,mu_y), f(1,1,mu_x,mu_y), f(2,1,mu_x,mu_y), f(3,1,mu_x,mu_y), f(4,1,mu_x,mu_y),
        f(-4,0,mu_x,mu_y), f(-3,0,mu_x,mu_y), f(-2,0,mu_x,mu_y), f(-1,0,mu_x,mu_y), f(0,0,mu_x,mu_y), f(1,0,mu_x,mu_y), f(2,0,mu_x,mu_y), f(3,0,mu_x,mu_y), f(4,0,mu_x,mu_y),
        f(-4,-1,mu_x,mu_y), f(-3,-1,mu_x,mu_y), f(-2,-1,mu_x,mu_y), f(-1,-1,mu_x,mu_y), f(0,-1,mu_x,mu_y), f(1,-1,mu_x,mu_y), f(2,-1,mu_x,mu_y), f(3,-1,mu_x,mu_y), f(4,-1,mu_x,mu_y),
        f(-4,-2,mu_x,mu_y), f(-3,-2,mu_x,mu_y), f(-2,-2,mu_x,mu_y), f(-1,-2,mu_x,mu_y), f(0,-2,mu_x,mu_y), f(1,-2,mu_x,mu_y), f(2,-2,mu_x,mu_y), f(3,-2,mu_x,mu_y), f(4,-2,mu_x,mu_y),
        f(-4,-3,mu_x,mu_y), f(-3,-3,mu_x,mu_y), f(-2,-3,mu_x,mu_y), f(-1,-3,mu_x,mu_y), f(0,-3,mu_x,mu_y), f(1,-3,mu_x,mu_y), f(2,-3,mu_x,mu_y), f(3,-3,mu_x,mu_y), f(4,-3,mu_x,mu_y),
        f(-4,-4,mu_x,mu_y), f(-3,-4,mu_x,mu_y), f(-2,-4,mu_x,mu_y), f(-1,-4,mu_x,mu_y), f(0,-4,mu_x,mu_y), f(1,-4,mu_x,mu_y), f(2,-4,mu_x,mu_y), f(3,-4,mu_x,mu_y), f(4,-4,mu_x,mu_y)
    ]
    return np.array(x)

#creates an eye postion array (laid out in write-up) where "1" will appear dx and dy from the origin 
def createEyeArray(dx, dy):
    center = 40
    empty = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    empty[center+dx+(9*-dy)] = 1
    return empty

#creates a data set of given size
def createData(size):
    retinal_data = np.empty([size, 81])
    eye_data = np.empty([size, 81])
    targets = np.empty([size, 81])
    for i in range(0, size):
        mu_x = random.randint(-4, 4)
        mu_y = random.randint(-4, 4)
        if mu_x >= 0:
            dx = random.randint(-4, 4-mu_x)
        else:
            dx = random.randint(-4-mu_x, 4)
        if mu_y >= 0:
            dy = random.randint(-4, 4-mu_y)
        else:
            dy = random.randint(-4-mu_y, 4)
        retinal_data[i] = createBivariateArray(mu_x, mu_y)
        eye_data[i] = createEyeArray(dx, dy)
        targets[i] = createBivariateArray(mu_x+dx, mu_y+dy)
    return retinal_data, eye_data, targets

#fabrication of training and testing data
training_data = createData(5000)
retinal_data_training = training_data[0]
eye_data_training = training_data[1]
targets_training = training_data[2]

testing_data = createData(500)
retinal_data_test = testing_data[0]
eye_data_test = testing_data[1]
targets_test = testing_data[2]

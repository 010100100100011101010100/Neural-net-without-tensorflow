#flow -> importing important modules -> readin data from the csv file -> spiltting the data into training and testing data -> creating functions for forward,backward propagation,relu,softmax,init parameters,update parameters,prediction,accuracy,gradient descent -> running the net -> creating test functions using matplotlib 

#importing all the required modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the data 
data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data=np.array(data)
m,n=data.shape

#dev data set
data_dev=data[1:1000].T
X_dev=data_dev[1:n]
Y_dev=data_dev[0]
X_dev=X_dev/255

#creating training set
data_train=data[1000:m].T
Y_train=data_train[0]
X_train=data_train[1:n]
X_train=X_train/255
_,m_train=X_train.shape



#creating functions required for the neural network
def init_params():
    W1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    W2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return W1,W2,b1,b2

def relu(x):
    return np.max(0,x)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def relu_der(x):
    return x>0


def forward(W1,W2,b1,b2,X):
    Z1=X.dot(W1)+b1
    A1=relu(Z1)
    Z2=A1.dot(W2)+b2
    A2=softmax(Z2)
    return Z1,Z2,A1,A2

def one_hot(Y):#one hot encoding the labels to subtract the value from their probabilities
    one=np.zeros((Y.size,Y.max()+1))
    one[np.arange(Y.size),Y]=1
    one=one.T
    return one

def backward(Z1,Z2,A1,A2,Y,X,W1,W2): 
    one=one_hot(Y)   
    dZ2=A2-one
    dW2=1/m*dZ2.dot(A1.T)
    db2=1/m*np.sum(dZ2)
    dZ1=W2.T.dot(dZ2)*relu_der(Z1)
    dW1=1/m*dZ1.dot(X.T)
    db1=1/m*np.sum(dZ1) 
    return dW1,dW2,db1,db2

def update_params(W1,W2,b1,b2,dW1,dW2,db1,db2,a):
    W1=W1-a*dW1
    b1=b1-a*b1
    b2=b2-a*b2
    W2=W2-a*dW2
    return W1,W2,b1,b2

def pred(A2):
    return np.argmax(A2,0)

def accuracy(pred,Y):
    print(pred,Y)
    return np.sum(Y==pred)/Y.size

def gradient_descent(iterations,a,X,Y):
    W1,W2,b1,b2=init_params()
    for i in range(iterations):
        Z1,Z2,A1,A2=forward(W1,W2,b1,b2,X)
        dW1,dW2,db1,db2=backward(Z1,Z2,A1,A2,W1,W2,X,Y)
        W1,W2,b1,b2=update_params(W1,W2,b1,b2,dW1,dW2,db1,db2,a)
        if i%50==0:
            print(i)
            prediction=pred(A2)
            print("The accuracy of the model at  ",i,"th iteration is ",accuracy(prediction,Y))
    return W1,b1,W2,b2
                  


#creating test functions using matplotlib

def get_pred(X, W1, b1, W2, b2):
    _,_,_,A2=forward(W1,W2,b1,b2,X)
    prediction=pred(A2)
    return prediction

def test_result(index, W1, b1, W2, b2):
    current=X_train[:,index,None]
    prediction=get_pred(current,W1,b1,W2,b2)
    label=Y_train[index]
    print(prediction)
    print(label)
#visualizing our pixelated digit 
    current=current.reshape((28,28))*255
    plt.gray()
    plt.imshow(current,interpolation='nearest')
    plt.show()

#testing part of the model with digit visualization
W1,b1,W2,b2=gradient_descent(500,0.2,X_train,Y_train)
for i in range(0,100):
    index=int(input())
    test_result(index,W1,b1,W2,b2)


    














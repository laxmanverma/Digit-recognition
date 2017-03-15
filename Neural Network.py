import numpy as np
import csv

'''
np.atleast_2d => converts provided data into 2-d array
np.random.random((x,y)) => creates and (x X y) matrix of random values between 0 and 1
np.shape(x) => return matrix configuration of x
'''

'''defining activation functions'''
def logistic(x):
    return (1/(1+pow(2.718281,(-1*x))))

def logistic_derivative(x):
    y = logistic(x)
    return (y * (1 - y))

def tanh(x):
    return (np.tanh(x))

def tanh_derivative(x):
    return (1.0 - np.tanh(x)**2)

'''defining NeuralNetwork class '''
class NeuralNetwork:
    def __init__(self,layers,activation="logistic"):
        self.activation=logistic
        self.activation_derivative=logistic_derivative
        if(activation=="tanh"):
            self.activation=tanh
            self.activation_derivative=tanh_derivative
        self.weights=[]
        for i in range(0,(len(layers)-2)):
            self.weights.append(((2*np.random.random((layers[i]+1,layers[i+1]+1)))-1)) # generate weights between -1 to 1
        i+=1
        self.weights.append(((2*np.random.random((layers[i]+1,layers[i+1])))-1)*0.25) # bias not included for last layer

    def train(self,x,y,lrate=0.2,epochs=10000):
        x=np.atleast_2d(x)
        temp = np.ones([x.shape[0], x.shape[1]+1]) #create a temp array matrix of ones of 20000*(784+1)
        temp[:, 0:-1] = x  # adding the bias unit to the input layer
        x = temp
        y=np.array(y)
        for k in range(0,epochs):
            i = np.random.randint(x.shape[0]) #randomly select an input array
            a=[x[i]]                            #a now stores the random input array from large pool of training data
            ''' now feeding forward
                initially a stores the first input matrix with bias
                each time it is multiplied with the corresponding weight matrix and resulting matrix is appended to a
            '''
            for l in range(0,len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            error=y[i]-a[-1] #error is the difference between result and expected output
            ''' now backpropagating
                starting from the last layer we compute its contribution compute delta for it and store it in delta matrix
            '''
            deltas=[error*self.activation_derivative(a[len(a)-1])] #last layer
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_derivative(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += lrate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

x=[]
y=[]

with open('train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    count=0
    for i in reader:
        count+=1
        a=[0,0,0,0,0,0,0,0,0,0]
        a[int(i['label'])]=1
        y.append(a)
        b=[]
        for j in range(0,784):
            b.append(i['pixel'+str(j)])
        x.append(b)
        if(count==20000):
            break

nn = NeuralNetwork([784,30,30,30,10],'tanh')
nn.train(x,y,lrate=0.08,epochs=100000)

with open('train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    count=0
    v=0
    for i in reader:
        count+=1
        a=[0,0,0,0,0,0,0,0,0,0]
        a[int(i['label'])]=1
        b=[]
        for j in range(0,784):
            b.append(i['pixel'+str(j)])
        x=a.index(max(a))
        y=np.argmax(nn.predict(b))
        if(x==y):
            v+=1
        print("Result "+str(y)+" Expected "+str(x)+" Status "+str(v)+"/"+str(count))
        if(count==40000):
            break
    print("Efficiency "+str((v*100)/count))


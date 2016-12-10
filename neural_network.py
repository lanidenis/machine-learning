
import numpy as np
import sys

#import matplotlib.pyplot as mpl

def sigmoid(inp):         
    return 1.0 / (1.0 + np.exp(-inp))
    
def d_sigmoid(inp):         
    return sigmoid(inp)*(1-sigmoid(inp))
    
class Neural_Net(object):
    
    def __init__(self, nInput, nHidden, nOutput):
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        #we only appoint bias to hidden and output layers
        self.bias = [np.random.randn(a, 1) for a in [nHidden,nOutput]]

        #matrix of 2D weight matrixes to store weights per node of input/hidden layers
        self.weight = [np.random.randn(a, b) for (a, b) in [(nInput, nHidden),(nHidden, nOutput)]]

    def feed(self, sample):
        for (b, w) in zip(self.bias, self.weight):
            temp = np.asarray([np.dot(w[:,i], sample) for i in range(w.shape[1])])
            temp = temp.reshape((temp.shape[0], 1))
            sample = sigmoid(temp + b)
        ##print "sample", sample
        return sample
    
    #backpropagation 
    def back_prop(self, x, y):
        #create empty starter matrices to hold gradient 
        bMat = [np.zeros(b.shape) for b in self.bias]
        wMat = [np.zeros(w.shape) for w in self.weight]
        
        #feed forward
        a = x
        sigma_out = [x]
        sigma_in = []
        
        #feed forward
        #level = 0
        for (b,w) in zip(self.bias, self.weight):
            #if level == 1: break
            ##print np.dot(w[:,1],a)
            i_n = np.asarray([np.dot(w[:,i], a) for i in range(w.shape[1])])
            i_n = i_n.reshape((i_n.shape[0], 1))
            ##print "i_n", i_n.shape, "level", level
            ##print "b", b.shape
            i_n = i_n + b
            ##print "i_n + b", i_n.shape, "level", level
            assert(len(i_n) == len(b))
            sigma_in.append(i_n) #save inputs for calculating bias gradient
            a = sigmoid(i_n) 
            sigma_out.append(a)#save outputs for calculating weights gradient 
            #level = level + 1
        
        #outer deltas (bias first, weights from bias)
        ##print "sigma_out[-1]", sigma_out[-1].shape
        ##print "sigma_in[-1]", sigma_in[-1].shape
        ##print "y", y.shape
        
        delta = (sigma_out[-1] - y)*d_sigmoid(sigma_in[-1])
        bMat[-1] = delta
        
        wMat[-1] = np.dot(sigma_out[-2], delta.transpose())
        ##print "wMat should be 30x10", wMat[-1].shape
        
        #inner deltas (bias as composite of previous weights/biases)
        
        #error -shapes (10,30) and (10,30) below
        ##print "delta", delta.shape
        ##print "weight", weight[-1].shape
        ##print "sigma_in", sigma_in[-2].shape
    
        #should be 30x1 !!!
        delta = np.dot(self.weight[-1], delta)*d_sigmoid(sigma_in[-2])
        bMat[-2] = delta 
        
        #should be 784x30!!
        wMat[-2] = np.dot(sigma_out[-3].reshape((784,1)), delta.transpose())  
        #wMat[-2] = [np.dot(delta, sigma_out[-3].transpose()[i]) for i in range(len(sigma_out))] 
        ##print "wMat2 should be 784,30", wMat[-2].shape
        
        return (bMat, wMat)
    
    
    #stochastic gradient descent 
    def master_run(self, train_data, test_data, epochs, mini, learn_rate):
        accuracies = []
        
        #stochastic gradient descent
        for epch in range(epochs):
            
            #randomly shuffle data
            np.random.shuffle(train_data)
            
            batch_index = 0  
            
            while batch_index < len(tr_x):#for each batch of size mini
                
                #extract next batch 
                batch = train_data[batch_index:batch_index + mini]
                
                #reset (zero-fill) bias and weight matrices
                bMat = [np.zeros(b.shape) for b in self.bias]
                wMat = [np.zeros(w.shape) for w in self.weight]
                
                for (x, y) in batch:
                    #retrieve gradient w.r.t cost for a each sample
                    (d_b, d_w)= self.back_prop(x, y)
                    #running sum of bias/weight error
                    bMat = [a+b for a,b in zip(bMat, d_b)]
                    wMat = [a+b for a,b in zip(wMat, d_w)]
                    
                #update weights/biases by subtracting average of running sum
                #calculations are fast because of matrix form
                self.bias = [b - (learn_rate/mini)*run_sum for b, run_sum in zip(self.bias, bMat)]
                self.weight = [w - (learn_rate/mini)*run_sum for w, run_sum in zip(self.weight, wMat)]
                
                batch_index = batch_index + mini
                ##print "Batch Index", batch_index
            
            #print "Completed Epoch", epch
            
            #test learned net on test sample
            output = [(self.feed(x).argmax(),y.argmax()) for (x, y) in test_data]
            num_correct = sum(int(x==y) for (x,y) in output)
            #print [x for (x, y) in output[0:10]]
            #print "num_correct", num_correct
            accuracy = 1.0*num_correct/len(test_data)
            #print "accuracy", accuracy 
            accuracies.append(accuracy)
            
            if (epch == epochs-1):
                np.savetxt(sys.argv[7], np.asarray(zip(*output)[0]), fmt='%i', delimiter=',')
        
        #plot accuracies vs epochs
        #mpl.plot(np.asarray(range(1, epochs+1)), np.asarray(accuracies)) 
        #mpl.axis([1, epochs, 0.0, 1.0])
        #mpl.ylabel('accuracy')
        #mpl.xlabel('epoch')
        #mpl.show()

"""
#parameters for testing small neural net
epochs = 1
mini = 2
learn_rate = 0.1

tr_x = np.array([[0.1, 0.1], [0.1, 0.2]])
tr_y = np.array([[1.0, 0.0], [0.0, 1.0]])
"""

#convert single digit output to column vector
def toVector(v):
    y = np.zeros((10,1))
    y[v] = 1
    return y 
    
tr_x = np.loadtxt(sys.argv[4],dtype='float',delimiter=',')
raw_y = np.loadtxt(sys.argv[5],dtype='int',delimiter=',')
tr_y = [toVector(digit) for digit in raw_y]

assert(len(tr_x) == len(tr_y))

test_x = np.loadtxt(sys.argv[6],dtype='float',delimiter=',')
raw_y2 = np.loadtxt('TestDigitY.csv.gz',dtype='int',delimiter=',')
test_y = [toVector(digit) for digit in raw_y2]

train_data = zip(tr_x, tr_y) #list of tuples (x,y)
test_data = zip(test_x, test_y) #list of tuples (x,y)

##print [sum(x) for (x, y) in test_data[0:10]]

#print "done loading"

mynet = Neural_Net(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
mynet.master_run(train_data, test_data, 30, 20, 3.0)

"""
for i in range(5):
    mynet = Neural_Net(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    mynet.master_run(train_data, test_data, 30, 20, 0.001*10**i)

mynet = Neural_Net(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
mynet.master_run(train_data, test_data, 30, 1, 3.0)
    
mynet = Neural_Net(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
mynet.master_run(train_data, test_data, 30, 5, 3.0)

mynet = Neural_Net(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
mynet.master_run(train_data, test_data, 30, 10, 3.0)

mynet = Neural_Net(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
mynet.master_run(train_data, test_data, 30, 20, 3.0)

mynet = Neural_Net(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
mynet.master_run(train_data, test_data, 30, 100, 3.0)
"""
    

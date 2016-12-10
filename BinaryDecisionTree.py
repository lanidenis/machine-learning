import sys
import csv
import math

train_x = []
test_x = []
train_y = []

with open(sys.argv[1], 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train_x.append(row)
        
with open(sys.argv[2], 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train_y.append(row[0])
        
with open(sys.argv[3], 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_x.append(row)

#for i in xrange(len(test_x)):
#    #print train_y[i]

if len(train_x) == 0 or len(train_y) == 0 or len(test_x) == 0:
    print "wait"
    
#print "TrainX" + str(len(train_x))
#print "TrainY" + str(len(train_y))
#print "TestX" + str(len(test_x))

class Node:
    def __init__(self, attr, thres, left, right):
        self.r = right
        self.l = left
        self.a = attr
        self.t = thres
        
class BuildTree:
    
    def __init__(self):
        self.height = 0
        self.nodes = 0
        self.leaves = 0
        
    def IG(self, data, label): 
            #finding largest IG is equal to finding lowest H(Y|X)
        ncol = len(data[0]) 
        nrow = len(data)
        
        cond_min = sys.float_info.max
        attr = 0
        th = 0
        flag = 0
        
        for i in range(ncol): 
            #sort the values for feature i
            values = []
            thresholds = []
            for j in range(nrow):
                values.append(float(data[j][i]))
            values.sort() 
            
            #compute possible thresholds for attr i
            for j in range(1, nrow):
                thresholds.append((values[j] - values[j-1])/2)
            
            #if (nrow == 25): #print thresholds
            
            #find optimal threshold and cond entropy for attr i
            #set as solution if cond entropy is smallest so far
            for j in range(nrow-1): 
                sum_g = 0.0
                sum_l = 0.0
                ones = 0.0
                zeroes = 0.0
                
                for k in range(nrow):
                    #print data[k][i] + " " + str(thresholds[j])
                    if float(data[k][i]) > thresholds[j] :               
                        sum_g = sum_g + 1
                        if int(label[k]) == 1: 
                            ones = ones + 1
                    else:
                        sum_l = sum_l + 1
                        if int(label[k]) == 0: 
                            zeroes = zeroes + 1

                #if(nrow == 25):
                #    #print str(sum_g) + " " + str(sum_l)

                #compute conditional entropy
                px = sum_g/nrow
                if sum_g != 0: py_x = ones/sum_g
                else: continue; #py_x == 0
                
                px0 = sum_l/nrow
                if sum_l != 0: py_x0  = zeroes/sum_l
                else: continue; #py_x0 = 0
                
                if py_x == 0: exp1 = 0
                else: exp1 = (py_x)*math.log(py_x,2)
                    
                if py_x == 1: exp2 = 0
                else: exp2 = (1-py_x)*math.log((1-py_x),2)
                
                if py_x0 == 0: exp3 = 0
                else: exp3 = (py_x0)*math.log(py_x0,2)
                
                if py_x0 == 1: exp4 = 0
                else: exp4 = (1-py_x0)*math.log((1-py_x0),2)
                
                cond = (-1)*(px)*(exp1 + exp2) + \
                        (-1)*(px0)*(exp3 + exp4)
                
               # #print str(cond) + " " + str(cond_min)
                
                #update attribute and threshold if necessary
                if cond < cond_min: 
                    flag = 1
                    attr = i
                    th = thresholds[j]
                    cond_min = cond
                    #if(nrow == 25): #print "attr: " + str(i) + "th: " + \
                    #                                str(thresholds[j])
                    #print "Hi" + i + th
               
        #compute entropy of y
        #p_zero = count/nrow
        #H_Y = - (p_zero*math.log(p_zero,2) + (1-p_zero)*math.log(1-p_zero,2))
        
        #compute conditional entropy of y | x
        
        if flag != 1: return [-1, -1]
        else: return [attr, th]
                    
        
    def GrowTree(self, data, label):         
        nrow = len(data)
        #print nrow
        
        if nrow == 0:
            return None 
        ncol = len(data[0])
          
        #if all outputs same return node that says "predict that output"
        flag = 0
        for i in range(1, nrow):
            if (label[i] != label[i-1]):
                flag = 1
                break
        if (flag == 0): 
            #print "outputs are all " + str(label[0])
            self.leaves = self.leaves + 1
            return Node(int(label[0]), None, None, None)
        
        #if all inputs the same then return node that says 
        #"predict the majority"
        #flag = 0
        #for i in range(1, nrow):
        #    for j in range(ncol):
        #        if data[i][j] != data[i-1][j] :
        #            flag = 1
        #            break
        #if flag == 0: 
        #    if sum(label)/ncol >= 0.5: return Node(1, None, None, None)
        #    else: return Node(0, None, None, None)
            
        #else split and grow tree
        result = self.IG(data, label)
        #print result
        
        data_l = []
        label_l = []
        data_r = [] 
        label_r = []
        
        if (result[0] == -1): #return percent probability, if group could
            #not be split up using cond entropy 
            my_sum = 0
            for i in range(len(label)):
                my_sum = my_sum + int(label[i])
                
            if float(my_sum)/float(nrow) > 0.5: 
                self.leaves = self.leaves + 1
                return Node(1, None, None, None)
            else: 
                #print "ZERO LEAF"
                self.leaves = self.leaves + 1
                return Node(0, None, None, None)
                
        for j in range(nrow):
            if (float(data[j][result[0]]) < result[1]):
                data_l.append(data[j])
                label_l.append(label[j])
            else:
                data_r.append(data[j])
                label_r.append(label[j])
                
        #if (nrow == 25): 
            #print result[0]
            #print result[1]
            #print data_r[1:5]
            #print label_r
        #print "data_l: " + str(len(data_l)) + "data_r: " + str(len(data_r))  
        
        self.nodes = self.nodes + 1      
        return Node(result[0], result[1], 
                    self.GrowTree(data_l, label_l), self.GrowTree(data_r, label_r))

        
build = BuildTree()
tree = build.GrowTree(train_x, train_y)
#print "Number of Nodes: " + str(build.nodes)
#print "Number of Leaves: " + str(build.leaves)

def traverse(tree, row):
    if tree == None : return None 
    
    if (tree.t == None) : #we've found leaf node
        #print str(tree.a)
        return tree.a
    
    if row[tree.a] < tree.t :  #compare to threshold
        return traverse(tree.l, row)
    else :
        return traverse(tree.r, row)

        
                    #predict labels for test_x and 
                    #write to a csv file
                    #then read in actual results and compare
                    #to those
predict_y = []
test_y = []

for i in xrange(len(test_x)): 
    predict_y.append(traverse(tree, test_x[i]))

#write to PredictY.csv
with open('PredictY.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for i in xrange(len(test_x)):
        writer.writerow([predict_y[i]])
        
with open('TestY.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_y.append(row[0])

summ = 0.0
for i in xrange(len(test_x)):
    if int(predict_y[i]) != int(test_y[i]) : 
        summ = summ + 1.0
    
#print "failure rate for test set: " + str(summ/len(test_x))

                    #predict labels for train_x and 
                    #compare to existing train_y
                    #then write to a csv file

predict_x = []

for i in xrange(len(train_x)): 
    predict_x.append(traverse(tree, train_x[i]))
    
summ = 0.0
for i in xrange(len(train_x)):
    if int(predict_x[i]) != int(train_y[i]) : 
        summ = summ + 1.0
    
#print "failure rate for trainin set: " + str(summ/len(train_x))

#write to Predict_Train.csv
with open('Predict_Train.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for i in xrange(len(train_x)):
        writer.writerow([predict_x[i]])    

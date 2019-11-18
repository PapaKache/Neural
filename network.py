import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
ARRAY1 = 1
ARRAY2 = 2

class Net:
    def __init__(self):
        self.relu_vector = np.vectorize(pyfunc=self.relu)
        self.relu_prime_vector = np.vectorize(pyfunc=self.relu_prime)        
        return
    def saveWeight(self,w,name):
        if os.path.exists(name):
            os.remove(name)

        row,col = w.shape
        f = open(name, 'w')
        s = ''
        for i in range(row):
            for j in range(col):
                v = str(w[i][j])
                s += v
                s += ','       
            s += '\n'            #print (s)

        f.write(s)
        f.close()
        return
    def loadWeight(self,path):
        f = open(path)
        lines=''
        for line in f.readlines():
            line = line.strip('\n')
            if len(line) > 1:
                lines += line
            
        f.close()

        res = []
        arr = lines.split(',')
        #print (arr)
        for i in range(len(arr)):
            if arr[i] == '':
                continue
            #print (arr[i])
            f = float(arr[i])
            res.append(f)
        return res

    def relu(self,x):
        if x >0:
            return x
        else:
            return 0

    def relu_prime(self,x):
        if x > 0.0:
            return 1.0
        else:
            return 0.0

    def softmax(self,Z):
        for i in range(len(Z)):
            m = np.max(Z[i])
            a = np.exp(Z[i] - m)
            Z[i] = a / np.sum(a)
        return Z
    
    def inference(self,w1,w2,inputdata,label):
        A1 = np.dot(inputdata,w1)
        A1 = self.relu_vector(A1)
        #out 100->10
        A2 = np.dot(A1,w2)
        #print(A2))
        ok = 0
        s = []
        for i in range(len(label)):
            v = np.argmax(A2[i])

            #print ('--------------->')
            #for k in range(49):
            #    print(A2[i][k])
            #print ('<---------------')

            v += 1
            s.append(v)
            if v == label[i]:
                ok += 1

        percent = ok / len(label)
        return percent,A2
    def train(self,w1,w2,inputdata, levels):
        size = len(inputdata)
        Z1 = np.dot(inputdata,w1)
        A1 = self.relu_vector(Z1)

        Z2 = np.dot(A1,w2)
        A2 = self.softmax(Z2)

        Delta2 = A2 - levels
        dW2 = np.dot(A1.T, Delta2)/size
        Delta1 = np.dot(Delta2,w2.T) * self.relu_prime_vector(Z1)
        dW1 = np.dot(inputdata.T, Delta1)/size
        return dW1,dW2
#28 * 28

'''
    x = 0
    lr = 0.1
    ldate,lvalue,cnt = loadData('data.csv')
    datelevels =  getDateLevels(ldate)
    #print (datelevels)
    valuelevels = getValueLevels(lvalue)
    for t in range (100):
        for i in range(0,cnt,32):
            start = i
            end   = start + 32

            dw1,dw2 = net.train(weight1,weight2,datelevels[start:end],valuelevels[start:end])
            #print(dw2)
            #exit()
            weight1 -= lr * dw1
            weight2 -= lr * dw2
            #if (i + 1)%100 == 0 :
            #    lr = lr - lr/10
            print('epoch:{0} '.format(i), end = "")
            p,vs = net.inference(weight1,weight2,datelevels,lvalue)
            #net.saveWeight(weight2,'w2-save.csv')
            print(p)

    net.saveWeight(weight1,'w1-save.csv')
    net.saveWeight(weight2,'w2-save.csv')
'''


'''
v = list(range(32))
    p,vs = net.inference(weight1,weight2,datelevels,v)
    #print('infer:{0} '.format(i), end = "")
    print(p)
    print(vs)
'''

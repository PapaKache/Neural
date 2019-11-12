import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
ARRAY1 = 1
ARRAY2 = 2

#28 * 28
def loadData(path):
    f = open(path)
    lines=''
    for line in f.readlines():
        line = line.strip('\n')
        line += ','
        if len(line) > 1:
            lines += line
        
    f.close()

    date = []
    value = []
    cnt = 0
    arr = lines.split(',')
    #print (arr)
    #print (arr)
    for i in range(len(arr)):
        if arr[i] == '':
            continue
        v = int(arr[i])
        if i%2 == 0:
            date.append(v)
        else:
            value.append(v)
        cnt+= 1
    return date,value,int(cnt/2)

def getDateLevels(labels):
    cnt = len(labels)
    arr = np.zeros((cnt, 64))
    for i in range(cnt):
        t = labels[i]
        for j in range(64):
            v = (t >> j) & 0x1
            arr[i][j] = v

    return arr


def getValueLevels(labels):
    cnt = len(labels)
    arr = np.zeros((cnt, 49))
    for i in range(cnt):
        idx = labels[i]
        idx -= 1
        arr[i][idx] = 1
    return arr


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
            v += 1
            s.append(v)
            if v == label[i]:
                ok += 1

        percent = ok / len(label)
        return percent,s
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

if __name__ == "__main__":
    net = Net()
    #w1 = np.random.normal(0,1,(64,365))
    #w2 = np.random.normal(0,1,(365,49))
    #net.saveWeight(w1, 'w1-6-random.csv')
    #net.saveWeight(w2, 'w2-6-random.csv')

    bw1 = net.loadWeight('w1-save.csv')
    bw2 = net.loadWeight('w2-save.csv')
    #list
    lw1 = list(bw1)
    lw2 = list(bw2)
    ldate,lvalue,cnt = loadData('test.csv')
    #print (cnt)
    #print (ldate)
    #print (lvalue)
    #array
    weight1 = np.array(lw1)
    weight2 = np.array(lw2)
    datelevels =  getDateLevels(ldate)
    #print (datelevels)
    valuelevels = getValueLevels(lvalue)

    

    weight1 = weight1.reshape(64,365)
    weight2 = weight2.reshape(365,49)

    x = 0
    lr = 0.1
    '''
    for t in range (60):
        for i in range(0,32,4):
            start = i
            end   = start + 4

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
    v = list(range(32))
    p,vs = net.inference(weight1,weight2,datelevels,v)
    #print('infer:{0} '.format(i), end = "")
    print(p)
    print(vs)


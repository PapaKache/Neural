import os
import time
import numpy as np
import matplotlib.pyplot as plt
ARRAY1 = 1
ARRAY2 = 2

#28 * 28
def loadData(path):
    f = open(path, 'rb')
    data = f.read()
    image = data[0:4]
    number = data[4:8]
    rows = data[8:12]
    cols = data[12:16]
    print('image:{0}'.format(image))
    print('number:{0}'.format(number))
    print('rows:{0}'.format(rows))
    print('cols:{0}'.format(cols))

    data = data[16:]
    f.close()
    return data

def loadLabel(path):
    f = open(path, 'rb')
    data = f.read()
    f.close()
    return data
   
def getLevels(labels):
    cnt = len(labels)
    arr = np.zeros((cnt, 10))
    for i in range(cnt):
        idx = labels[i]
        arr[i][idx] = 1
    return arr



def showImage(data):
    #im = struct.unpack_from('>784B',data)
    ldata = list(data)
    im = np.array(ldata)
    im = im.reshape(28,28)
    fig = plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()

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
    bw1 = net.loadWeight('w1-random.csv')
    bw2 = net.loadWeight('w2-random.csv')
    btestdata = loadData('t10k-images.idx3-ubyte')
    #list
    lw1 = list(bw1)
    lw2 = list(bw2)
    ltestdata = list(btestdata)

    for i in range(len(ltestdata)):
        if ltestdata[i] > 128:
            ltestdata[i] = 1
        else:
            ltestdata[i] = 0
    #array
    weight1 = np.array(lw1)
    weight2 = np.array(lw2)
    testdata = np.array(ltestdata)
    testdata = testdata.reshape(10000,784)

    weight1 = weight1.reshape(784,100)
    weight2 = weight2.reshape(100,10)
    testlabels = loadLabel('test_label.bin')
    testlevels = getLevels(testlabels)

    x = 0
    lr = 0.1
    for i in range(2000):

        if i > 1500:
            lr = 0.05
        elif i > 1000:
            lr = 0.07
        x += 64
        if (x > 10000 - 64):
            x = 0
        start = x 
        end   = start + 64
        dw1,dw2 = net.train(weight1,weight2,testdata[start:end],testlevels[start:end])
        #print(dw2)
        #exit()
        weight1 -= lr * dw1
        weight2 -= lr * dw2
        #if (i + 1)%100 == 0 :
        #    lr = lr - lr/10
        print('epoch:{0} '.format(i), end = "")
        p,vs = net.inference(weight1,weight2,testdata,testlabels)
        #net.saveWeight(weight2,'w2-save.csv')
        print(p)



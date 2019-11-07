import time
import numpy as np
import matplotlib.pyplot as plt
ARRAY1 = 1
ARRAY2 = 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    if x >0:
        return x
    else:
        return 0

def relu_prime(x):
    if x > 0.0:
        return 1.0
    else:
        return 0.0

def getMaxIndex(ls):
    m = -98987
    idx = 0
    for i in range(len(ls)):
        if m <= ls[i]:
            idx = i
            m = ls[i]

    return idx

relu_vector = np.vectorize(pyfunc=relu)
relu_prime_vector = np.vectorize(pyfunc=relu_prime)

def loadWeight(path):
    f = open(path)
    lines=''
    for line in f.readlines():
        line = line.strip('\n')
        lines += line
    f.close()

    res = []
    arr = lines.split(',')
    for i in range(len(arr)):
        #print (arr[i])
        f = float(arr[i])
        res.append(f)
    return res
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
   
def inference(w1,w2,inputdata,label):
    A1 = np.dot(inputdata,w1)
    A1 = relu_vector(A1)
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

def getLevel(idx):
    nl = np.zeros((1,10))
    nl[0][idx] = 1
    #print(nl[0])
    return nl
'''
def softmax(arr):
    #for i in range(len(arr)):
    a = np.exp(arr - arr.max())
    b = np.sum(a)
    return a/b
'''
def softmax(Z):
    for i in range(len(Z)):
        m = np.max(Z[i])
        a = np.exp(Z[i] - m)
        Z[i] = a / np.sum(a)
    return Z
def getLevels(labels):
    cnt = len(labels)
    arr = np.zeros((cnt, 10))
    for i in range(cnt):
        idx = labels[i]
        arr[i][idx] = 1
    return arr

def train(w1,w2,inputdata, levels):
    #input middle output
    #middle [1 x 784] * [784 x 100]->[1 x 100]
    #print('len inputdata:{0}'.format(len(inputdata)))
    #print(inputdata[0])
    #print(levels)
    size = len(inputdata)
    Z1 = np.dot(inputdata,w1)
    A1 = relu_vector(Z1)

    Z2 = np.dot(A1,w2)
    A2 = softmax(Z2)

    Delta2 = A2 - levels
    dW2 = np.dot(A1.T, Delta2)/size
    Delta1 = np.dot(Delta2,w2.T) * relu_prime_vector(Z1)
    dW1 = np.dot(inputdata.T, Delta1)/size
    return dW1,dW2

def showImage(data):
    #im = struct.unpack_from('>784B',data)
    ldata = list(data)
    im = np.array(ldata)
    im = im.reshape(28,28)
    fig = plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()

#
#bytes
bw1 = loadWeight('w1-random.csv')
bw2 = loadWeight('w2-random.csv')
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
#print(testdata[0:2])
#print(testlevels[0:2])
x = 0
lr = 0.1
for i in range(10000):
    x += 64
    if (x > 10000 - 64):
        x = 0
    start = x 
    end   = start + 64
    dw1,dw2 = train(weight1,weight2,testdata[start:end],testlevels[start:end])
    #print(dw2)
    #exit()
    weight1 -= lr * dw1
    weight2 -= lr * dw2
    if (i + 1)%100 == 0 :
        lr = lr - lr/10

    p,vs = inference(weight1,weight2,testdata,testlabels)
    print(p)



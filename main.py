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
#reference book function
def inference2(w1,w2,inputdata):
    A0 = inputdata.reshape(784,1)
    print (A0.shape())
    #X = W * I
    A1 = np.dot(w1 , A0)
    A1 = relu_vector(A1)
    A2 = np.dot(w2 , A1)
    print (A2)
    
def inference(w1,w2,inputdata):
    #input middle output

    #middle 784->100
    '''
    inputdata          w1                              mid
    [1 2 .... 784] * [1 2 .... 100
                      2 3 .....100          ===> [1 ....... 100] 1行100列
                      ...
                      784 ........
    
                     ]

    '''
    inputdata = inputdata.reshape(1,784)
    #print(tmp)
    A1 = np.dot(inputdata,w1)
    A1 = relu_vector(A1)
    '''
    midf                w2                          out
    [1 .........100] *[1 2 ....10
                       2 3 ....10         
                       ..........              ====>[1......10] 1行10列
                       100 .....10
                       ]
    '''
    #out 100->10
    A2 = np.dot(A1,w2)
    A2 = A2.reshape(1,10)
    l = A2.tolist()
    #print(l[0])
    return getMaxIndex(l[0])

def getLevel(idx):
    nl = np.zeros((1,10))
    nl[0][idx] = 1
    #print(nl[0])
    return nl

def softmax(arr):
    #for i in range(len(arr)):
    a = np.exp(arr - arr.max())
    b = np.sum(a)
    return a/b

def train(w1,w2,inputdata, label):
    #input middle output

    #middle [1 x 784] * [784 x 100]->[1 x 100]
    inputdata = inputdata.reshape(1,784)
    Z1 = np.dot(inputdata,w1)
    A1 = relu_vector(Z1)

    #out [1 x 100] * [100 * 10]->[1 x 10]
    Z2 = np.dot(A1,w2)
    A2 = softmax(Z2)
    #print(Z2)
    #print(A2)
    idx = label[0]
    T2 = getLevel(idx)
    #[1 x 10] - [1 x 10] = [1 x 10]
    Delta2 = A2 - T2
    #[100 x 1] * [1 x 10] = [100 x 10]
    dW2 = np.dot(A1.T, Delta2)
    Delta1 = np.dot(Delta2,w2.T)
    dW1 = np.dot(inputdata.T, Delta1) * relu_prime_vector(Z1)
    return dW1,dW2

def showImage(data):
    #im = struct.unpack_from('>784B',data)
    ldata = list(data)
    im = np.array(ldata)
    im = im.reshape(28,28)
    fig = plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()

def inference_test_images(w1,w2, data,labels,start,end):
    starttime = time.time()
    cnt = 10000
    ok = 0
    percent = 0
    for i in range(start,end):
        label = labels[i]
        start =  i * 784
        r = inference(w1,w2, data[start:start+784])
        if r == label:
            ok +=1
        percent = ok/(i + 1)
        #if (i + 1)% (cnt/20) == 0:
        #    print('[0-{1}]inference accuracy:{0}'.format('%.3f'%percent, i))
    
    endtime = time.time()
    t = endtime - starttime
    print('inference accuracy:{0},time:{1}(s)'.format('%.3f'%percent,'%.2f'%t))

def train_test_images(w1,w2,data,labels,start,end):
    for i in range(start,end):
        imgstart = i * 784
        dW1,dW2 = train(w1,w2, data[imgstart:imgstart+784], labels[i:i+1])
        dW2 = 0.05 * dW2
        dW1 = 0.05 * dW1
        w2 = w2 - dW2
        w1 = w1 - dW1
        #inference_test_images(w1,w2, data,labels,0,1000) 
     
    print ('train {0}-{1}'.format(start, end))
    return w1,w2
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

weight1 = weight1.reshape(784,100)
weight2 = weight2.reshape(100,10)
#for i in range(10):
#    start =  i * 784
#    r = inference(w1,w2, testdata[start:start+784])
#    print ('inference result:{0}'.format(r))
#showImage(testdata[start:start+784])
testlabels = loadLabel('test_label.bin');
start = 0
end = 10000

for i in range(100):
        #print(weight2)
    inference_test_images(weight1,weight2,testdata,testlabels,0,10000)
    start = 0
    end   = 10000
    w1,w2 = train_test_images(weight1,weight2,testdata,testlabels,start,end)
    weight2 = w2
    weight1 = w1
























'''
def loadWeight(path,array):
    f = open(path)
    lines=''
    for line in f.readlines():
        line = line.strip('\n')
        lines += line
    f.close()

    arr = []
    wa = lines.split(',')
    print ('number')
    print (len(wa))
    for i in range(0,len(wa),2):
        #print (i)
        #print(wa[i])
        #print(wa[i + 1])
        #print ('---------------------------')
        v1 = int(wa[i])
        v2 = int(wa[i+1])
        if array == ARRAY1:
            w = v1 - v2
        else: 
            w = v2 - v1
        
        if w < 0:
            w = -1
        elif w == 0:
            w = 0
        else:
            w = 1
        arr.append(w)
    #print(arr[1])
    return arr

def saveWeight(w,name,arr):
    f = open(name, 'w')
    s = ''
    for i in range(len(w)):
        v = str(w[i])
        s += v
        s += ','
        if arr == ARRAY1:
            if (i + 1) % 100 == 0:
                s += '\n'
        else:
            
            if(i + 1) % 10 == 0:
                s += '\n'
                #print (s)

    f.write(s)
    f.close()
    return



'''



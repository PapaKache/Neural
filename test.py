import numpy as np
ls = [1,2,3,4,5,6]
ar = np.array(ls)
m = np.argmax(ar)
t = ar.reshape(2,3)
print(m)
print('------------------')
print(len(t))
'''
def fun(a):
    a[0] = 123

c = np.array((1,1))
print(c)
fun(c)
print(c)
a = np.random.normal(0.5,1,784 * 100)
print (len(a))
f = open('w1-random.csv','w')
s = ''
for i in range(len(a)):
    s += str(a[i])
    s += ','
    if (i + 1) % 100 == 0:
        s += '\n'
        f.write(s)
        s = ''
f.close()
'''

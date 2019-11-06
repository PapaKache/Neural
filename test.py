import numpy as np

def fun(a):
    a[0] = 123

c = np.array((1,1))
print(c)
fun(c)
print(c)
'''
a = np.random.normal(0.5,1,100 * 10)
print (len(a))
f = open('w2-random.csv','w')
s = ''
for i in range(len(a)):
    s += str(a[i])
    s += ','
    if (i + 1) % 10 == 0:
        s += '\n'
        f.write(s)
        s = ''
f.close()
'''

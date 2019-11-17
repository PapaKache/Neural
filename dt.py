
file = open('t.csv', 'rb')
w = open('data.csv', 'w')
lines=''
i = 0
for line in file.readlines():
    i = i + 1
    line = line.decode('utf-8')
    line = line.strip('\n')
    line = line.replace(',','/')
    d = line.split('/')
    v0 = str('%02d'%int(d[0]))
    v1 = str('%02d'%int(d[1]))
    v2 = str('%02d'%int(d[2]))
    date = v0 + v1 + v2
    value = str(d[3])
    w.write(date)
    w.write(',')
    w.write(value)
    w.write('\n')






w.close()
file.close()

# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""
import sys
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow

from Ui_main import Ui_MainWindow
from network import Net
import numpy as np
import _thread as thread
from PyQt5.QtCore import pyqtSignal

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

def sort(A2):
    size = len(A2)
    #print(A2.shape)
    #print(size)
    t = list(range(49))
    for i in range(size):
        for j in range(49):
            t[j] =(j + 1, A2[i][j]) 
        t = sorted(t,  key = lambda v:  v[1],  reverse = True)
        #print (t)
        #print('------------------------------------------------------------')
    return t
    
class MainWindow(QMainWindow, Ui_MainWindow):
    finsh_signal  = pyqtSignal()
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.net = Net()
   
        bw1 = self.net.loadWeight('w1-save.csv')
        bw2 = self.net.loadWeight('w2-save.csv')
        lw1 = list(bw1)
        lw2 = list(bw2)
        weight1 = np.array(lw1)
        weight2 = np.array(lw2)
        self.weight1 = weight1.reshape(64,365)
        self.weight2 = weight2.reshape(365,49)
        
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(100)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.valueChanged.connect(self.valuechange)
        self.progressBarPercent.setValue(0)
        self.progressBarAcc.setValue(0)
        self.trainning = False
        self.finsh_signal.connect(self.callFinish)
        
        
    def threadTrain(self, ls):
        self.trainning = True
        batch_size = 32
        times =ls
        lr = 0.1
        ldate,lvalue,datacnt = loadData('data.csv')
        datelevels =  getDateLevels(ldate)
        #print (datelevels)
        valuelevels = getValueLevels(lvalue)
        progress = 0
        acc =  0
        cnt = int (datacnt/batch_size)*batch_size
        rem = datacnt % batch_size
        for t in range (times):
            
            for i in range(0,cnt,batch_size):
                start = i
                end   = start + batch_size
                dw1,dw2 = self.net.train(self.weight1,self.weight2,datelevels[start:end],valuelevels[start:end])
                self.weight1 -= lr * dw1
                self.weight2 -= lr * dw2
                p,vs = self.net.inference(self.weight1,self.weight2,datelevels,lvalue)
    
                cal = ( (t*datacnt + i + 1))/ (times * datacnt)
                cal = int(cal * 100)
                #print('{0}*{1}+{2}/({3} * {4})={5}'.format(t, datacnt, i, times,  datacnt, cal))
                if progress  != cal:
                    progress = cal
                    self.progressBarPercent.setValue(progress)
                    acc = int(p * 100)
                    self.progressBarAcc.setValue(acc)
            if rem >0:
                start += batch_size
                end   = start + rem
                #print('{0},{1}'.format(start, end))
                dw1,dw2 = self.net.train(self.weight1,self.weight2,datelevels[start:end],valuelevels[start:end])
                self.weight1 -= lr * dw1
                self.weight2 -= lr * dw2
                p,vs = self.net.inference(self.weight1,self.weight2,datelevels,lvalue)
    
                cal = ( (t*datacnt + i  + 1))/ (times * datacnt)
                cal = int(cal * 100)
                #print('{0}*{1}+{2}/({3} * {4})={5}'.format(t, datacnt, i, times,  datacnt, cal))
                if progress  != cal:
                    progress = cal
                    self.progressBarPercent.setValue(progress)
                    acc = int(p * 100)
                    self.progressBarAcc.setValue(acc)
                
        #self.progressBarPercent.setValue(100)
        
        self.net.saveWeight(self.weight1,'w1-save.csv')
        self.net.saveWeight(self.weight2,'w2-save.csv')
        self.finsh_signal.emit()
        #
        self.trainning = False
    def callFinish(self):
        self.progressBarPercent.setValue(100)
        self.textEditResult.setText('训练完成')
        
    @pyqtSlot()
    def on_pushButtonTrain_clicked(self):
        self.textEditResult.setText('开始训练')
        if self.trainning == True:
            self.textEditResult.setText('正在训练')
            return
        times = self.horizontalSlider.value()
        #self.labelTime.setText(str(times))
        print(times)
        thread.start_new_thread( self.threadTrain, (times, ))
       
    @pyqtSlot()
    def on_pushButtonInference_clicked(self):
        date = self.calendarWidget.selectedDate()
        t = str(date.toPyDate())
        l = t.split('-')
        sdate = str(l[0]) + str(l[1]) + str(l[2])
        print (sdate)
        ldata = [int(sdate)]
        lvalue = [0]
        datelevels =  getDateLevels(ldata)
        #valuelevels = getValueLevels(lvalue)
        p,a2 = self.net.inference(self.weight1,self.weight2,datelevels,lvalue)
        data = sort(a2)
        s = ''
        for i in range(49):
            n, p = data[i]
            s += str(n)
            s += ','
        self.textEditResult.setText(s)
            
    def valuechange(self):
        times = self.horizontalSlider.value()
        self.labelTime.setText(str(times))

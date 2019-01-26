import numpy as np
import pandas as pd
import tensorflow as tf
import queue

np.random.seed(1)
tf.set_random_seed(1)

class Environment:
    def __init__(self):
        self.ev = []    #事件表
        self.en = []    #事件
        self.q = {'m1':queue.Queue(),'m2':queue.Queue(),'m3':queue.Queue(),'m4':queue.Queue(),'o1':queue.Queue(),'o2':queue.Queue(),'o3':queue.Queue()} #7个出口队列，每10秒出去一个人
        self.nowTime = 0
        self.totalTime = 0

        #定义模拟参数
        self.exitPersonNum = 0  #已经出去的人数
        self.building = {'x':1000,'y':1000}
        self.personNum = 1000
        self.person = []    #定义100个人
        self.mainEntrances = [{'x':0,'y':0},{'x':1000,'y':0},{'x':0,'y':1000},{'x':1000,'y':1000}]
        self.otherEntrances= [{'x':500,'y':0},{'x':500,'y':1000},{'x':1000,'y':500}]
        
        for i in range(self.personNum):
            randomX = (1000-0)*np.random.random_sample() + 0
            randomY = (1000-0)*np.random.random_sample() + 0
            #类型为1,2,3  分别代表正常游客，团体游客，残疾游客
            p_type = round((3-1)*np.random.random_sample()+1)
            p_speed = p_type*10
            location = {'x':randomX,'y':randomY,'type':p_type,'speed':p_speed,'action':0,'arrived':0,'exit':0}
            self.person.append(location)
        # print(self.person)
        # print(self.mainEntrances)
        # print(self.q)
    
    def getDis(self,x1,y1,x2,y2):
        return np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

    #得到第t个people的状态
    def getState(self,t):
        if t >= 1000 or t < 0:
            return 'input error!'
        #得到t people的坐标
        x = self.person[t]['x']
        y = self.person[t]['y']
        speed = self.person[t]['speed']
        #得到t people到各出口的距离以及时间
        mainEntrancesDis = []
        mainEntrancesTime = []
        otherEntrancesDis = []
        otherEntrancesTime = []
        for i in range(4):
            dis = self.getDis(x,y,self.mainEntrances[i]['x'],self.mainEntrances[i]['y'])
            mainEntrancesDis.append(dis)
            mainEntrancesTime.append(round(dis/speed))
        for i in range(3):
            dis = self.getDis(x,y,self.otherEntrances[i]['x'],self.otherEntrances[i]['y'])
            otherEntrancesDis.append(dis)
            otherEntrancesTime.append(round(dis/speed))
        
        #得到各个出口的等待时间
        mainEntrancesWaitTime = []
        otherEntrancesWaitTime = []
        for i in range(1,5):
            mainEntrancesWaitTime.append(self.q['m'+str(i)].qsize() * 10)
        for i in range(1,4):
            otherEntrancesWaitTime.append(self.q['o'+str(i)].qsize() * 10)

        ret_state = [x,y] + mainEntrancesDis + otherEntrancesDis + mainEntrancesTime + otherEntrancesTime + mainEntrancesWaitTime + otherEntrancesWaitTime
        return ret_state

    #得到reward
    def getReward(self,t,action):
        #得到t people的坐标
        x = self.person[t]['x']
        y = self.person[t]['y']
        speed = self.person[t]['speed']
        #得到出口的等待事件和路程时间
        Xe = 0
        Ye = 0
        gama = 0            #安全因素
        EntranceName = ''
        if action >= 1 and action <=4:  #主出口
            Xe = self.mainEntrances[action-1]['x']
            Ye = self.mainEntrances[action-1]['y']
            gama = 0
            EntranceName = 'm'+str(action)
        else:                           #其它出口
            Xe = self.otherEntrances[action-1-4]['x']
            Ye = self.otherEntrances[action-1-4]['y']
            EntranceName = 'o'+str(action-4)
            gama = -5

        Twait = self.q[EntranceName].qsize() * 10
        Tdis = round(self.getDis(x,y,Xe,Ye)/speed)
        reward = -(Twait + Tdis) + gama
        return reward



    def doAction(self,t,action):
        #action 1-7对应走哪个出口
        self.person[t]['action'] = action
        # self.run()

    #run 1秒钟
    def run(self):
        #更新每个人的位置
        for i in range(self.personNum):
            #对action不为0且未达到出口的用户进行更新
            if self.person[i]['action']!=0 and self.person[i]['arrived']==0:
                x = self.person[i]['x']
                y = self.person[i]['y']
                action = self.person[i]['action']
                speed = self.person[i]['speed']
                #得到相应出口的位置
                Xe = 0
                Ye = 0
                EntranceName = ''
                if action >= 1 and action <=4:  #主出口
                    Xe = self.mainEntrances[action-1]['x']
                    Ye = self.mainEntrances[action-1]['y']
                    EntranceName = 'm'+str(action)
                else:                           #其它出口
                    Xe = self.otherEntrances[action-1-4]['x']
                    Ye = self.otherEntrances[action-1-4]['y']
                    EntranceName = 'o'+str(action-4)

                sin_ = (Ye-y)/self.getDis(x,y,Xe,Ye)
                cos_ = (Xe-x)/self.getDis(x,y,Xe,Ye)
                x_update = speed * 1 * cos_ + x
                y_update = speed * 1 * sin_ + y
                #判断边界
                if action >=1 and action <=4:
                    if x_update >= 1000:
                        x_update = 1000
                    if x_update <= 0:
                        x_update = 0
                    if y_update >= 1000:
                        y_update = 1000
                    if y_update <= 0:
                        y_update = 0
                else:
                    if y_update <= 0:
                        x_update = 500
                        y_update = 0
                    if y_update >=1000:
                        x_update = 500
                        y_update = 1000
                    if x_update >=1000:
                        x_update = 1000
                        y_update = 500
                self.person[i]['x'] = x_update
                self.person[i]['y'] = y_update

                #判断是否到达出口
                if self.person[i]['x'] == Xe and self.person[i]['y'] == Ye:
                    self.person[i]['arrived'] = 1
                    #丢进出口队列里
                    #定义队列事件
                    tempEv = {'personId':i,'waitTime':10}
                    self.q[EntranceName].put(tempEv)
        
        #更新队列
        for j in range(1,8):
            EntranceName = ''
            if j <= 4:
                EntranceName = 'm'+str(j)
            else:
                EntranceName = 'o'+str(j-4)
            
            if not self.q[EntranceName].empty():    
                tempEv = self.q[EntranceName].get()
                tempEv['waitTime'] -= 1
                if tempEv['waitTime'] == 0:    #代表等待完毕    成功出去
                    self.person[tempEv['personId']]['exit'] = 1
                    self.exitPersonNum += 1
                else:   #否则，只是更新队列元素
                    self.q[EntranceName].put(tempEv)
                    for k in range(self.q[EntranceName].qsize()-1):
                        self.q[EntranceName].put(self.q[EntranceName].get())

        #时间更新
        self.nowTime += 1
        if self.exitPersonNum == self.personNum:
            return 1
        return 0


    def personDeparture(self):
        return 0

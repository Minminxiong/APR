# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:00:16 2019

@author: XMM
"""
from random import sample

class Cascade:
    def __init__(self, k, cascade):       
        self.k = k
        self.N = 595460
        self.isLargerK = False
        self.cascade_with_unique_node = None
        self.uniqueAK = None
        self.uniqueAK_with_time_without_hashtag = None
        
        """去除重复的节点及其时间，获取最终无重复节点的级联"""
        cascade_list = cascade.strip("\n").split(" ")       
        cascade_with_unique_node = [cascade_list[0]]             
        for i in range(2, len(cascade_list), 2):
            if cascade_list[i] not in  cascade_with_unique_node:
                cascade_with_unique_node.append(cascade_list[i - 1])
                cascade_with_unique_node.append(cascade_list[i])

        for i in range(1, len(cascade_with_unique_node)):
            cascade_with_unique_node[i] = int(cascade_with_unique_node[i])
            

        if len(cascade_with_unique_node) > 2*k + 1:
            self.isLargerK = True
            self.cascade_with_unique_node = cascade_with_unique_node
        
        
        #self.cascadeString = cascade


#        self.first_forwarder_features = None
#        self.firstK_forwarder_features = None
#        self.temporal_features = None
#        self.structure_features = None
        
    #当cascade_list的长度大于2 * k时，才能运行下面的函数
    def get_cascade_series_subcas(self):
        

        #获取带时间的前k个节点信息，不带话题
        self.uniqueAK_with_time_without_hashtag = self.cascade_with_unique_node[1: 2 * self.k + 1]
    
        cas_without_time = [self.cascade_with_unique_node[i] for i in range(2,len(self.cascade_with_unique_node),2)]
        cas_without_time.insert(0, self.cascade_with_unique_node[0])
        #前k个感染节点列表，不带话题
        self.uniqueAK = cas_without_time[1:self.k+1]
        #return cas_without_time
        
    def getTargetNodes(self, flag = 0):
        '''
        flag为0：只将被感染的节点作为目标节点
        为1： 不被感染的作为目标节点，挑选和感染数目相近的节点数
        为2：感染+不感染作为目标节点
        '''
        infected_f = open("infected.txt","w")
        if flag == 0:
            target = self.cascade_with_unique_node[self.k + 1:]
            target_nodes0 = [node for node in list(target) if node < self.N]
            return target_nodes0
        elif flag==1:
            target = set(range(self.N)) - set(self.cascade_with_unique_node[1:])
            target_nodes1 = sample(list(target), len(self.cascade_with_unique_node[self.k + 1:]))
            return target_nodes1
        else:
            target = self.cascade_with_unique_node[self.k + 1:]
            target_nodes0 = [node for node in list(target) if node < self.N]
            print("级联中被感染的节点数:",len(target_nodes0))
            target = set(range(self.N)) - set(self.cascade_with_unique_node[1:])
            target_nodes1 = sample(list(target), len(target_nodes0))
            print("挑选的未被感染的节点数:",len(target_nodes1))
            infected_f.write(str(len(target_nodes0)) + " " + str(len(target_nodes1)))
            infected_f.close()
            return target_nodes0 + target_nodes1
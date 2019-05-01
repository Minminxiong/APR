# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:48:38 2019

@author: XMM
"""
from pyrwr.rwr import RWR
import time
import networkx as nx
import time
#import networkx as nx
import numpy as np
from pickleClass import Pickle
from concurrent.futures import ProcessPoolExecutor

def read_embeddings_dict(file):
    embeddings_dict = {}
    with open(file, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            line = [float(v) for v in line]
            embeddings_dict[line[0]] = line[1:]
    return embeddings_dict


def get_G_nodeToHashtag():
    # 读文件获取G和nodeToHashtag数据
    start = time.time()

    nodeToH_pkl = Pickle("../dataset/nodeTohashtag.pkl")
    G_pkl = Pickle("../dataset/G.pkl")
    print("开始获取每个节点参与的话题数......")
    p_nodeToH = nodeToH_pkl.read_pickel()
    print("每个节点参与话题数获取完成......")

    print("构建网络......")
    p_G = G_pkl.read_pickel()
    print("网络构建完成......")
    #    pprint.pprint(p_nodeToH)
    print('读取文件时间: {}'.format(time.time() - start))
    return p_nodeToH, p_G

class Features():
    #rwr是类变量，所有类的实例共享该变量
    start = time.time()
    rwr = RWR()
    rwr.read_graph("../dataset/tab_follower_gcc.edgelist", "undirected")
    print("生成rwr模型花费的时间：", time.time() - start)
    print("读取网络结构嵌入特征......")
    topology_embeddings_file = "../embeddings/follower_gcc.anony.embeddings"
    embeddings_dict = read_embeddings_dict(topology_embeddings_file)
    print("读取网络embedding特征数据花费的时间：", time.time() - start)
    nodeToHashtag, G =  get_G_nodeToHashtag()

    def __init__(self, unique_Ak,hashtag):
        self.N = 595460
        self.unique_Ak = unique_Ak
        self.hashtag = hashtag

    def the_early_adopter_shortest(self, target_node):
        """取早期感染节点中和目标节点距离最短的节点和路径长度"""
#        shortest_path_length = 1000000
        path_length = [nx.shortest_path_length(self.G, source = node, target = target_node) for node in self.unique_Ak]
        min_path_length = min(path_length)
        shortest_early_adopter = self.unique_Ak[path_length.index(min_path_length)]
#        for node in firstKV:
#            path_length = nx.shortest_path_length(G, source = node, target = target_node)      
#            if path_length < shortest_path_length:
#                shortest_path_length = path_length
#                shortest_early_adopter = node
        return shortest_early_adopter, min_path_length
    
    
    def get_adoption_p(self, target_node):
        #belta is a damping factor, empirically set as 0.05
        belta = 0.05 
        shortest_early_adopter, shortest_length = self.the_early_adopter_shortest(target_node)
        #print("shortest_early_adopter:", shortest_early_adopter)
        try:
            target_node_hashtags = self.nodeToHashtag[target_node]
        except:
            target_node_hashtags = set()   
            return 0
        if self.hashtag in target_node_hashtags:
            return 1
            
        try:
            shortest_early_adopter_hashtags = self.nodeToHashtag[shortest_early_adopter]  
            #判断shortest_early_adopter是否参与过话题，找出参与的话题集；若没参与话题，则设置其话题集为空
        except:
            shortest_early_adopter_hashtags = set()
            return 0
#        print("new_hashtag:", self.new_hashtag)
#        print("shortest_early_adopter_hashtags:", shortest_early_adopter_hashtags)
#        print("target_node_hashtags:", target_node_hashtags)

        
        adoption_p = (belta**shortest_length) * len(shortest_early_adopter_hashtags & target_node_hashtags)/len(shortest_early_adopter_hashtags | target_node_hashtags)
        return adoption_p


    def First_Forwarder_Features(self):
        '''
        outdeg_v1: the degree of first forwarder in G
        num_hashtags_v1: number of past hashtags used by v1
        orig_connections_k:number of early 2 to k forwarders who are friends of the first forwarder
        '''
#        unique_Ak = self.unique_AK
        First_Forwarder_id = self.unique_Ak[0]   #取第一个被感染的节点的编号
        outdeg_v1 = self.G.degree(First_Forwarder_id)  #第一个被感染节点的度
        num_hashtags_v1 = len(self.nodeToHashtag[First_Forwarder_id])    #第一个被感染结点参与的话题数
        orig_connections_k = len(set(self.G.neighbors(First_Forwarder_id))  & set(self.unique_Ak))  #早期K个感染节点中是第一感染节点的邻居节点数
    #    First_Forwarder_features_name = ["outdeg_v1", "num_hashtags_v1", "orig_connections_k"]
        first_forwarder_features = [outdeg_v1, num_hashtags_v1, orig_connections_k]        
        return first_forwarder_features
    
    
    def First_K_Forwarders_Features(self):
        '''
        计算前K个被感染节点的相关特征
        '''
        #前K个节点中有不同用户的个数（同一用户可能会多次参与）
        unique_Ak_num = len(self.unique_Ak)   
        #Ak_deg_G，DegreeView类型 早期感染节点集AK在大网络G中的度
        Ak_deg_G = [self.G.degree(node) for node in self.unique_Ak]
        
        #前K个早期感染节点的总的一阶邻居数
        views_1k = sum(Ak_deg_G)  
        #构建前K感染节点的子图
        subG_Ak = self.G.subgraph(self.unique_Ak)  
        Ak_deg_subG = [subG_Ak.degree(node) for node in self.unique_Ak]
        max_, min_, ave = [max(Ak_deg_subG), min(Ak_deg_subG), sum(Ak_deg_subG)/len(Ak_deg_subG)]  #子图中节点的最大度、最小度、平均度
        subG_edges = subG_Ak.size()     #子图中的边数
        return [unique_Ak_num,  max_, min_, ave,  views_1k, subG_edges]
    
    def Structure_Features(self):
        
        Ak_deg_G = []
        for node in self.unique_Ak:
            if node < self.N:
                Ak_deg_G.append(self.G.degree(node))
            else:
                Ak_deg_G.append(0)
    #    G_Ak_deg = G.degree(unique_Ak)
    #    Ak_deg_G = [G_Ak_deg[i] for i in range(len(unique_Ak))]
        max_AkG_deg, min_AkG_deg, ave_AkG_deg = [max(Ak_deg_G), min(Ak_deg_G), sum(Ak_deg_G)/len(Ak_deg_G)]
        return [max_AkG_deg, min_AkG_deg, ave_AkG_deg]
    
    def get_Temporal_Features(self, firstKV_with_time, K):
        '''
        获取前k个感染节点的时间相关特征
        '''
    
        firstKV = firstKV_with_time
        #前K个节点中有不同用户的个数（同一用户可能会多次参与）
        unique_Ak_num = len(self.unique_Ak)   
        #Ak_deg_G，DegreeView类型 早期感染节点集AK在大网络G中的度     
        
        Ak_deg_G = [self.G.degree(node) for node in self.unique_Ak]

            
        #前K个早期感染节点的总的一阶邻居数
        views_1k = sum(Ak_deg_G)  
        
        #保存前K个感染节点中，第i个到第1个参与话题的时间差
        
        time_1_i = [ (firstKV[2*i] - firstKV[0]) for i in range(1,K)]     
# =============================================================================
#         i = 1
#         time_1_i = []
#         while i < K:
#             temp_i = firstKV[ 2 * i ] - firstKV[0] 
#             time_1_i.append(temp_i)
#             i += 1
# =============================================================================
            
        #计算前k个感染节点中第i个和第i+1个感染时间差
        time_i_iplus1 = [(firstKV[2*(i+1)] - firstKV[2*i]) for i in range(1, K-1)]
# =============================================================================
#         i = 1
#         time_i_iplus1 = []   
#         for i in range(K-1):
#             interval_i = firstKV[2*(i+1)] - firstKV[2*i]
#             time_i_iplus1.append(interval_i)
# =============================================================================
        
        #前k个节点相邻两个节点间感染时间差平均值
        time_ave_k = sum(time_i_iplus1)/len(time_i_iplus1)   
        #前k/2个节点间感染时间差平均值
        time_ave_1_k2 = sum(time_i_iplus1[:K//2])/len(time_i_iplus1[:K//2])   
        #后k/2个节点间感染时间差平均值
        time_ave_k2_k = sum(time_i_iplus1[K//2+1:])/len(time_i_iplus1[K//2+1:])  
        
        #第1个和第k个感染时间差
        time_k = time_1_i[-1]    
        #信息被看见的速度
        speed_exposure = views_1k/time_k   
        #信息扩散的速度
        speed_adoption = unique_Ak_num/time_k  
        
        temporal_features = [time_ave_k, time_ave_1_k2, time_ave_k2_k, speed_exposure, speed_adoption] + time_1_i
#        temporal_features.extend(time_1_i)
        return temporal_features

    def calEuclideanDistance(self, vec1, vec2):
        '''计算两向量间的欧式距离'''
        dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
        #    round(dist, 6)
        return dist

    def get_reachability_features(self, target_node):
        '''
        adoption_node: int类型，早期感染中的一个节点
        target_node: int类型，待预测的目标节点
        distKV_target: list类型， 第一个元素记录的是目标节点编号，其他元素则是目标节点到早期节点的距离（从拓扑结构角度）
        '''
        #    start = time.time()

        #    distKV_target.append(target_node)
        target_topology_embedding = np.array(self.embeddings_dict[target_node])
        distKV_target = [self.calEuclideanDistance(np.array(self.embeddings_dict[adoption_node]), \
                                                   target_topology_embedding) for adoption_node in self.unique_Ak]
        # =============================================================================
        #         for adoption_node in self.unique_Ak:
        #             adoption_topology_embedding = np.array(self.embeddings_dict[adoption_node])
        #
        #             dist = self.calEuclideanDistance(adoption_topology_embedding, target_topology_embedding)
        #             distKV_target.append(dist)
        # =============================================================================

        #    end = time.time()
        #    print ('计算单个目标节点到早期感染节点的运行时间',end-start)
        return distKV_target
    
    def Target_Features(self, target_node):
        '''
        计算目标节点的特征
        '''
    #    target_features_names = ["target_neighbors_num","First_nbr_adopter",  \
    #                             "Second_nbr_adopter", "Third_nbr_adopter", \ 
    #                             "past_adoption_exist", "past_adoption_target_num",\
    #                             "com_hashtags_max", "com_hashtags_min", \
    #                             "com_hashtags_ave",]
        
        #计算early adopters到目标节点的probability of random walk with restarting
        
        adoption_p = self.get_adoption_p(target_node)
        r = self.rwr.compute(target_node)
        r_K = [r[node][0] for node in self.unique_Ak]
        distKV_target = self.get_reachability_features(target_node)
        target_1_neighbors = list(self.G.neighbors(target_node))
        #目标节点的邻居节点数
        target_neighbors_num = len(target_1_neighbors)  
        #目标节点邻居和早期感染节点的交集的个数
        First_nbr_adopter = len(set(target_1_neighbors) & set(self.unique_Ak))   
        
#        target_2_neighbors = [ for node in target_1_neighbors]
        target_2_neighbors = []
        for node in target_1_neighbors:
            target_2_neighbors.extend(list(self.G.neighbors(node)))
        #目标节点二阶邻居和早期感染节点的交集的个数
        Second_nbr_adopter = len(set(target_2_neighbors) & set(self.unique_Ak))  
        
        
        #判断目标用户之前是否参与过话题讨论，参与过 past_adoption_exist赋值为1， 未参与则为0   
        target_3_neighbors = []
        for node in target_2_neighbors:
            target_3_neighbors.extend(list(self.G.neighbors(node)))
            
        #目标节点二阶邻居和早期感染节点的交集的个数
        Third_nbr_adopter = len(set(target_3_neighbors) & set(self.unique_Ak))  
        
        try:
            past_adoption_target = self.nodeToHashtag[target_node]
            #目标节点参与的话题数,目标节点的活跃度
            past_adoption_target_num = len(past_adoption_target)  
            #目标节点参与话题的标签设为1
            past_adoption_exist = 1     
            common_hashtags = [len(set(self.nodeToHashtag[node]) & set(past_adoption_target))  for node in self.unique_Ak]
            com_hashtags_max, com_hashtags_min, com_hashtags_ave = \
            [max(common_hashtags), min(common_hashtags), sum(common_hashtags)/len(common_hashtags)]
        except:
            #目标节点从未参与话题讨论
            past_adoption_exist = 0
            #目标节点从未参与话题讨论，参与的话题数为0
            past_adoption_target_num = 0

            com_hashtags_max, com_hashtags_min, com_hashtags_ave = [0,0,0]

        # =============================================================================
        #         #过去节点参与过话题讨论，则计算各个早期感染节点和目标节点间话题讨论的重合个数
        #         if past_adoption_exist:
        #             common_hashtags = [len(set(self.nodeToHashtag[node]) & set(past_adoption_target))  for node in self.unique_Ak]
        #             com_hashtags_max, com_hashtags_min, com_hashtags_ave = \
        #             [max(common_hashtags), min(common_hashtags), sum(common_hashtags)/len(common_hashtags)]
        #
        #         else:
        #             com_hashtags_max, com_hashtags_min, com_hashtags_ave = [0,0,0]
        # =============================================================================
            
        target_features = [target_node, target_neighbors_num, First_nbr_adopter, Second_nbr_adopter, \
                           Third_nbr_adopter, past_adoption_exist, past_adoption_target_num, \
                           com_hashtags_max, com_hashtags_min, com_hashtags_ave]
        target_features += r_K
        target_features += distKV_target
        target_features += [adoption_p]
        return target_features

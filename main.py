# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:02:08 2019

@author: XMM
"""

import argparse
import time
from pickleClass import Pickle
from cascade import Cascade
from features import Features
#from pyrwr.rwr import RWR
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from functools import partial
import csv
import networkx as nx
from Log_Reg import Log_reg

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run APR.")

	parser.add_argument('--input', nargs='?', default='../dataset/timeline.train.right',
	                    help='Input cascadaes file path')
    
	parser.add_argument('--k', type=int, default=25,
	                    help='Number of early adopters. Default is 25.')

	parser.add_argument('--output', nargs='?', default='features/train_features.csv',
	                    help='Features path')
	parser.add_argument('--cascades_counts', type=int, default=0,
	                    help='calcute number of cascades')
	return parser.parse_args()


def main(args):
    start_start = time.time()
    feature_file = open(args.output, "w", newline='')
    feature_writer = csv.writer(feature_file)
    features_name = ["target_node", "target_neighbors_num", "First_nbr_adopter", "Second_nbr_adopter", \
                     "Third_nbr_adopter", "past_adoption_exist", "past_adoption_target_num", \
                     "com_hashtags_max", "com_hashtags_min", "com_hashtags_ave"] + ["outdeg_v1", "num_hashtags_v1", "orig_connections_k", "unique_Ak_num",  \
                     "max_subG", "min_subG", "ave_subG",  "views_1k", "subG_edges", "max_AkG_deg", \
                     "min_AkG_deg", "ave_AkG_deg", "time_ave_k", "time_ave_1_k2", "time_ave_k2_k", \
                     "speed_exposure", "speed_adoption"] + ["time_1_{}".format(i) for i in range(1, args.k)] + \
                     ["r_node_{}".format(i) for i in range(args.k)] + ["distKV_target_{}".format(i) for i in range(args.k)] + \
                     ["adoption_p", "label"]
                     
    feature_writer.writerow(features_name)
#    p_nodeToH, p_G =  get_G_nodeToHashtag()
#    p_G, p_nodeToH = {},{}
    count = 0
    with open(args.input, 'r', encoding = 'utf-8') as f:
        for line in f:
            if count > args.cascades_counts:
                break
            
            cas = Cascade(args.k, line)
            if cas.isLargerK:
                cas.get_cascade_series_subcas()
#                print(cas.uniqueAK)
                hashtag = cas.cascade_with_unique_node[0]
                
                features = Features(cas.uniqueAK, hashtag)
                
                start = time.time()
                
                Infected_targetNodes = cas.getTargetNodes(flag = 0)
                with ThreadPoolExecutor(max_workers = cpu_count()) as excutor:
                    tf_infected = excutor.map(features.Target_Features, Infected_targetNodes)
                    
                    
                targetNodes = cas.getTargetNodes(flag =1)
#                partial_target_features = partial(features.Target_Features())
                with ThreadPoolExecutor(max_workers = cpu_count()) as excutor:
                    tf = excutor.map(features.Target_Features, targetNodes)
                    
                fff = features.First_Forwarder_Features()
                fkff = features.First_K_Forwarders_Features()
                sf = features.Structure_Features()
                gtf = features.get_Temporal_Features(cas.uniqueAK_with_time_without_hashtag, cas.k)
#                tf = features.Target_Features(0, rwr)
                feature_list = fff + fkff + sf + gtf
                for tf_list in tf_infected:
                    feature_list1 = tf_list[: -2 * args.k -1] + feature_list + tf_list[-2* args.k -1:] + [1]
                    feature_writer.writerow(feature_list1)
                    
                for tf_list in tf:
                    feature_list1 = tf_list[: -2 * args.k -1] + feature_list + tf_list[-2* args.k -1:] + [0]
                    feature_writer.writerow(feature_list1)
#                print(fff, fkff, sf, gtf, tf)
                print("串行运行时间：", time.time() - start)
                
# =============================================================================
#                 start = time.time()
#                 with ThreadPoolExecutor(max_workers = cpu_count()) as excutor:
#                     fff = excutor.submit(features.First_Forwarder_Features)
#                     fkff = excutor.submit(features.First_K_Forwarders_Features)
#                     sf = excutor.submit(features.Structure_Features)
#                     gtf = excutor.submit(features.get_Temporal_Features,cas.uniqueAK_with_time_without_hashtag, cas.k)
#                     tf = excutor.submit(features.Target_Features,0, rwr)
#                     print(type(fff.result()))
#                     print(fff.result(),fkff.result(), sf.result(), gtf.result(),tf.result())
#                     print("并行运行时间：", time.time() - start)
# =============================================================================
                count += 1
    feature_file.close()
    print("总运行时间：", time.time()- start_start)


if __name__ == "__main__":

	args = parse_args()
	main(args)

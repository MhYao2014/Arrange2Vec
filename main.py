# -*- coding: utf-8 -*-
import sys
import pdb
import queue
import mosek
import random
import pickle
import numpy as np
from tqdm import tqdm
from sym2pos_save_arrange_as_sdp import mosek_solver, get_ele_order
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
from scipy.special import comb


def sub_local_mi_parser(path: str):
    word2id_dict = {}
    id2word_and_freq_dict = {}
    line_count = 0
    with open(path, 'r') as f:
        while True:  # 第一遍遍历语料，建立词表和word2id的映射
            line =f.readline()
            if len(line) == 0:
                break
            print(f"正在读取第{line_count}行.", end='\r')
            line_count += 1
            line = line.split('\t')
            w_id = line[0]
            word = line[1]
            word_freq = line[2]
            pij = line[3:]
            word2id_dict[word] = int(w_id)
            id2word_and_freq_dict[int(w_id)] = [word, float(word_freq)]
    f.close()
    line_count = 0
    sym_mat = np.zeros([len(word2id_dict), len(word2id_dict)])
    with open(path, 'r') as f:
        while True:  # 第二遍遍历语料，建立local_mi_am_dict
            line =f.readline()
            if len(line) == 0:
                break
            print(f"正在读取第{line_count}行.", end='\r')
            line_count += 1
            line = line.split('\t')
            w_id = line[0]
            w_id = int(w_id)
            word = line[1]
            word_freq = line[2]
            pij = line[3:]
            for col_id in range(w_id, len(word2id_dict)):
                local_mi_ij = float(pij[col_id])
                if col_id != w_id and local_mi_ij != 0:
                    sym_mat[w_id, col_id] = local_mi_ij  #记录local_mi
                    sym_mat[col_id, w_id] = sym_mat[w_id, col_id]
    return sym_mat, word2id_dict, id2word_and_freq_dict


def load_sym_mat(path: str, if_load: bool):
    if if_load is False:
        sym_mat, word2id_dict, id2word_and_freq_dict = sub_local_mi_parser(path)
        
        print(f'保存local_mi_sym_mat')
        np.save('./local_mi_sym_mat.npy', sym_mat)
        
        print(f'保存word2id_dict')
        with open('./word2id_dict.pkl', 'wb') as f:
            pickle.dump(word2id_dict, f)
        f.close()
        
        print(f'保存id2word_and_freq_dict')
        with open('./id2word_and_freq_dict.pkl', 'wb') as f:
            pickle.dump(id2word_and_freq_dict, f)
        f.close()
    else:
        print(f'载入local_mi_sym_mat')
        sym_mat = np.load('./local_mi_sym_mat.npy')
        
        print(f'载入word2id_dict')
        with open('./word2id_dict.pkl', 'rb') as f:
            word2id_dict = pickle.load(f)
        f.close()
        print(f'载入id2word_and_freq_dict')
        with open('./id2word_and_freq_dict.pkl', 'rb') as f:
            id2word_and_freq_dict = pickle.load(f)
        f.close()
    max_am = sym_mat.max()
    [dim, _] = sym_mat.shape
    delta_length = 10 / dim
    for w_id in range(len(word2id_dict)):
        sym_mat[w_id, w_id] = max_am + 10.5 - delta_length * (w_id+1)
    return sym_mat, word2id_dict, id2word_and_freq_dict


def get_normalized_freq(id2word_and_freq_dict):
    id_freq_list = [[item[0], item[1][1]] for item in id2word_and_freq_dict.items()]
    id_freq_list = sorted(id_freq_list, key=lambda item:item[0])
    total_count = sum([id_freq[1] for id_freq in id_freq_list])
    new_freq_list = []
    for idx, freq in id_freq_list:
        new_freq_list.append((freq/total_count)**0.75)
    modified_total_count = sum(new_freq_list)
    modified_freq_list = []
    for freq in new_freq_list:
        modified_freq_list.append(freq/modified_total_count)
    return modified_freq_list


def merge_local2global_emb(global_embedding, local_embedding, l_wid2g_wid, global_word_count):
    [local_emb_num, local_emb_dim] = local_embedding.shape
    [global_emb_num, global_emb_dim] = global_embedding.shape
    assert global_emb_dim == local_emb_dim
    # 将local_embedding旋转到大致对齐global_emb的对应部分
    l_id_in_g_list = sorted([l_wid2g_wid[l_wid] for l_wid in l_wid2g_wid])
    tmp_g_emb = global_embedding[l_id_in_g_list, :]
    U, S, V = np.linalg.svd(np.matmul(local_embedding.T, tmp_g_emb))
    local_embedding = np.matmul(local_embedding, np.matmul(U, V.T))
    
    for l_emb_idx in range(local_emb_num):
        g_emb_idx = l_wid2g_wid[l_emb_idx]
        g_weight = global_word_count[g_emb_idx] / (global_word_count[g_emb_idx] + 1)
        l_weight = 1 / (global_word_count[g_emb_idx] + 1)
        global_embedding[g_emb_idx] = g_weight * global_embedding[g_emb_idx, :] + l_weight * local_embedding[l_emb_idx, :]
        global_word_count[g_emb_idx] += 1
    return global_embedding, global_word_count


def save_global_emb(global_embedding, global_word_count, task_num):
    if task_num % 5 == 0:
        print('正在保存slow embedding，请不要打断')
        np.save('./slow_global_embed.npy', global_embedding)
        np.save('./slow_global_count.npy', global_word_count)
        print('slow embedding保存完成')
    np.save('./fast_global_embed.npy', global_embedding)
    np.save('./fast_global_count.npy', global_word_count)
    
    
def get_sub_mat(sym_mat, global_wid_list):
    sub_mat = sym_mat[global_wid_list, :][:, global_wid_list]
    ele_order_matrix = get_ele_order(sub_mat)  # [row_i, col_j, mat_val]
    l_wid2g_wid = {}
    for local_idx in range(len(global_wid_list)):
        l_wid2g_wid[local_idx] = global_wid_list[local_idx]
    return l_wid2g_wid, ele_order_matrix
    
    
def work(pid, jobs, result, sym_mat):
    while True:
        try:
            global_wid_list = jobs.get()  # worker尝试获取任务
            # if word_freq_wid is not None:  # worker只有在拿到任务时才执行任务
            if isinstance(global_wid_list, list):
                if isinstance(global_wid_list[0], int):
                    # 从sym_mat中按照global_wid_list取出子矩阵
                    l_wid2g_wid, ele_order_matrix = get_sub_mat(sym_mat, global_wid_list)
                    local_embedding = mosek_solver(ele_order_matrix, len(l_wid2g_wid))
                    result.put([local_embedding, l_wid2g_wid])  # 向主进程发送求解结果
            if global_wid_list == 'Stop':  # worker进程结束的条件：听到主进程说结束了
                return
        except queue.Empty:
            result.put(None)  # 向主进程发送求解结果


def multi_prod_single_consume(sym_mat:np.array, modified_freq_list:list, if_load_g_emb:bool, 
                              global_emb_path:str='./slow_global_embed.npy', 
                              global_count_path:str='./slow_global_count.npy'):
    jobs = Queue()
    result = JoinableQueue()
    num_of_cpu = cpu_count()
    mosek_solver_num = 4
    cpu_per_mosek_task = num_of_cpu // mosek_solver_num
    sub_mat_dim = 100
    if if_load_g_emb is False:
        global_embedding = np.zeros([len(modified_freq_list), sub_mat_dim])
        global_word_count = np.array([0 for i in range(len(modified_freq_list))])
    else:
        global_embedding = np.load(global_emb_path)
        global_word_count = np.load(global_count_path)
    # 子矩阵数量N需要足够大以近似原始的超大矩阵
    sub_mat_num = 50
    vocab_idx_list = list(range(len(modified_freq_list)))
    for sub_mat_count in range(sub_mat_num):
#         sub_mat_vocab = random.choices(vocab_idx_list, modified_freq_list, k=sub_mat_dim)
        sub_mat_vocab = random.choices(vocab_idx_list, k=sub_mat_dim)
        jobs.put(sub_mat_vocab)
        
    print('starting workers')
    [Process(target=work, args=(i, jobs, result, sym_mat)).start()
     for i in range(mosek_solver_num)]
    
    result_count = 0
    while True:
        try:
            r = result.get()  # 尝试获取结果，并且不阻塞等待结果
            if r is not None:
                result_count += 1
                # 将sub_mat融合到full_mat
                local_embedding, l_wid2g_wid = r
                global_embedding, global_word_count = merge_local2global_emb(
                    global_embedding, local_embedding, l_wid2g_wid, global_word_count
                )
                save_global_emb(global_embedding, global_word_count, result_count)
                print(f'Total sub mat:{sub_mat_num}; Solved sub mat:{result_count}.')
                result.task_done()
        except queue.Empty:  # 防止报错
            pass
        if result_count == sub_mat_num:  # 一旦所有任务数量达到上限，就可以结束任务了
            break
                
    for w in range(mosek_solver_num):
        jobs.put('Stop')
    
    result.join()
    jobs.close()
    result.close()
    

if __name__ == "__main__":
    # 载入对称矩阵
    sub_local_mi_path = './sub_local_mi.txt'
    if_load = True
    sym_mat, word2id_dict, id2word_and_freq_dict = load_sym_mat(sub_local_mi_path, if_load)
    # 开始抽样并求解
    modified_freq_list = get_normalized_freq(id2word_and_freq_dict)
    multi_prod_single_consume(sym_mat, modified_freq_list, if_load)
#     pdb.set_trace()

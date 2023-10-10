# -*- coding: utf-8 -*-
import sys
import pdb
import mosek
import random
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau
# from main import load_sym_mat
from sym2pos_save_arrange_as_sdp import mosek_solver

inf = 0.0


def generate_sym_mat(vocab_size: int, embedding_size: int):
    diag_ele_list = [vocab_size + (vocab_size**0.1)*random.uniform(1, vocab_size) for i in range(vocab_size)]
    diag_ele_list = sorted(diag_ele_list)
#     tmp_embedding = np.random.randn(vocab_size, embedding_size)
#     for i in range(vocab_size):
#         tmp_vec_norm = np.dot(tmp_embedding[i, :], tmp_embedding[i, :])**0.5
#         tmp_embedding[i, :] = tmp_embedding[i, :] / tmp_vec_norm * diag_ele_list[i]**0.5
#     sym_mat = np.matmul(tmp_embedding, tmp_embedding.T)
    
    sym_mat = np.zeros([vocab_size, vocab_size])
    for row_i in range(vocab_size):
        for col_j in range(row_i, vocab_size):
#             print(f'row:{row_i};col_j:{col_j}', end='\r')
            if row_i == col_j:
                sym_mat[row_i, col_j] = diag_ele_list[row_i]
            else:
                new_ele = (diag_ele_list[col_j]**0.5) * (diag_ele_list[row_i]**0.5) * random.uniform(-0.5, 0.5)
                sym_mat[row_i, col_j] = new_ele
                sym_mat[col_j, row_i] = sym_mat[row_i, col_j]
    return sym_mat


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def get_ele_order(sym_mat, if_load_new: bool = False):
    if if_load_new is not True:
        [vocab_size, _] = sym_mat.shape
        num_ele = int(vocab_size * (vocab_size + 1) / 2)

        print('开始填充ele_order_list')
        ele_order_list = np.zeros([num_ele, 3])    
        ele_idx_in_list = 0;
        for row_i in tqdm(range(vocab_size)):
            for col_j in range(row_i, vocab_size):
                ele_order_list[ele_idx_in_list, 0] = row_i
                ele_order_list[ele_idx_in_list, 1] = col_j
                ele_order_list[ele_idx_in_list, 2] = sym_mat[row_i, col_j]
                ele_idx_in_list += 1
        ele_order_list = ele_order_list[ele_order_list[:,2].argsort(),:]

        print('开始填充ij2order_id_dict')
        ij2order_id_dict = {}
        for order_idx in tqdm(range(num_ele)):
            key = '.'.join([str(int(ele_order_list[order_idx, 0])), str(int(ele_order_list[order_idx, 1]))])
            ij2order_id_dict[key] = order_idx

        np.save('./toy_ele_order_list.npy', ele_order_list)

        with open('./toy_ij2order_id_dict.pkl', 'wb') as f:
            pickle.dump(ij2order_id_dict, f)
        f.close()
    else:
        print(f'载入ele_order_list')
        ele_order_list = np.load('./toy_ele_order_list.npy')
        
        print(f'载入ij2order_id_dict')
        with open('./toy_ij2order_id_dict.pkl', 'rb') as f:
            ij2order_id_dict = pickle.load(f)
        f.close()
    
    return ele_order_list, ij2order_id_dict


def partly_get_linear_bound_idx(ele_order_list, ij2order_id_dict, 
                         current_word_idx, known_word_idx, total_vs):
    [ele_num, _] = ele_order_list.shape
    key = '.'.join([str(known_word_idx), str(current_word_idx)])
    order_id_of_ij = ij2order_id_dict[key]
    known_lb_idx = [None, None]
    known_ub_idx = [None, None]
    print('开始找下界', end='\r')
    for ele_order_idx in (range(order_id_of_ij, -1, -1)):
        row_i = int(ele_order_list[ele_order_idx, 0])
        col_j = int(ele_order_list[ele_order_idx, 1])
        # 根据row_i和col_j检查当该元素是不是已知的,
        # 有三种情况：
        if_known_val = False
        if row_i < current_word_idx and col_j < current_word_idx:
            if known_word_idx in [row_i, col_j]:
                if_known_val = True
                # 有对于越低的词频，越有概率不收录该约束
#                 if np.random.uniform(0, 1) > 0.1*current_word_idx/total_vs:
#                     if_known_val = False
        # col_j = current_word_idx and row_i < current_word_idx
        if col_j == current_word_idx and row_i < known_word_idx:
            if_known_val = True

        # 若是已知元素，那么需要区别是上界还是下界
        if if_known_val:
            if ele_order_idx < order_id_of_ij:
                known_lb_idx = [row_i, col_j]
                break

    print('开始找上界', end='\r')
    for ele_order_idx in (range(order_id_of_ij+1, ele_num-total_vs)):
        row_i = int(ele_order_list[ele_order_idx, 0])
        col_j = int(ele_order_list[ele_order_idx, 1])
        # 有三种情况：
        if_known_val = False
        # row_i, col_j < current_word_idx
        if row_i < current_word_idx and col_j < current_word_idx:
            if known_word_idx in [row_i, col_j]:
                if_known_val = True
                # 有对于越低的词频，越有概率不收录该约束
#                 if np.random.uniform(0, 1) > 0.1*current_word_idx/total_vs:
#                     if_known_val = False
        # col_j = current_word_idx and row_i < current_word_idx
        if col_j == current_word_idx and row_i < known_word_idx:
            if_known_val = True

        # 若是已知元素，那么需要区别是上界还是下界
        if if_known_val:
            if ele_order_idx > order_id_of_ij:
                known_ub_idx = [row_i, col_j]
                break
    return known_lb_idx, known_ub_idx


def get_linear_bound_idx(ele_order_list, ij2order_id_dict, 
                         current_word_idx, known_word_idx, total_vs):
    [ele_num, _] = ele_order_list.shape
    key = '.'.join([str(known_word_idx), str(current_word_idx)])
    order_id_of_ij = ij2order_id_dict[key]
    known_lb_idx = [None, None]
    known_ub_idx = [None, None]
    print('开始找下界', end='\r')
    for ele_order_idx in tqdm(range(order_id_of_ij, -1, -1)):
        row_i = int(ele_order_list[ele_order_idx, 0])
        col_j = int(ele_order_list[ele_order_idx, 1])
        # 根据row_i和col_j检查当该元素是不是已知的,
        # 有三种情况：
        if_known_val = False
        if row_i < current_word_idx and col_j < current_word_idx:
            if_known_val = True
        # col_j = current_word_idx and row_i < current_word_idx
        elif col_j == current_word_idx and row_i < known_word_idx:
            if_known_val = True

        # 若是已知元素，那么需要区别是上界还是下界
        if if_known_val:
            if ele_order_idx < order_id_of_ij:
                known_lb_idx = [row_i, col_j]
                break

    print('开始找上界', end='\r')
    for ele_order_idx in tqdm(range(order_id_of_ij+1, ele_num-total_vs)):
        row_i = int(ele_order_list[ele_order_idx, 0])
        col_j = int(ele_order_list[ele_order_idx, 1])
        # 有三种情况：
        if_known_val = False
        # row_i, col_j < current_word_idx
        if row_i < current_word_idx and col_j < current_word_idx:
            if_known_val = True
        # col_j = current_word_idx and row_i < current_word_idx
        elif col_j == current_word_idx and row_i < known_word_idx:
            if_known_val = True

        # 若是已知元素，那么需要区别是上界还是下界
        if if_known_val:
            if ele_order_idx > order_id_of_ij:
                known_ub_idx = [row_i, col_j]
                break
    return known_lb_idx, known_ub_idx


def get_linearBound(known_lb_idx, known_ub_idx,
                    current_word_idx, known_word_idx,
                    ij2order_id_dict, all_embedding, ele_order_list):
    [lb_row_i, lb_col_j] = known_lb_idx
    lb_val = []
    lb_A = []
    [ub_row_i, ub_col_j] = known_ub_idx
    ub_val = []
    ub_A = []
    if lb_row_i is not None:
        # 处理下界
        lb_row_i = int(lb_row_i)
        lb_col_j = int(lb_col_j)
        if current_word_idx not in [lb_row_i, lb_col_j]: # 下界是定值
            lb_val.append(np.dot(all_embedding[lb_row_i, :], all_embedding[lb_col_j, :]))
            lb_A = all_embedding[known_word_idx, :].tolist()
        else: # 
            another_vec_idx = lb_row_i
            lb_val.append(0)
            lb_A = (all_embedding[known_word_idx, :] - all_embedding[another_vec_idx, :]).tolist()
    if ub_row_i is not None:
        # 处理上界
        ub_row_i = int(ub_row_i)
        ub_col_j = int(ub_col_j)
        if current_word_idx not in [ub_row_i, ub_col_j]: # 上界是定值
            ub_val.append(np.dot(all_embedding[ub_row_i, :], all_embedding[ub_col_j, :]))
            ub_A = all_embedding[known_word_idx, :].tolist()
        else:
            another_vec_idx = ub_row_i
            ub_val.append(0)
            ub_A = (all_embedding[known_word_idx, :] - all_embedding[another_vec_idx, :]).tolist()
    return lb_val, lb_A, ub_val, ub_A


def get_pb(emb_dim, all_embedding, current_word_idx, ele_order_list, ij2order_id_dict):
    # 下面5个变量存放向量内积诱导的线性约束
    bkc = []
    blc = []
    buc = []
    asub = []
    aval = []
    # 下面4个变量存放当前向量的模长最小值约束x_1^2 + ... + x_n^2 >= 上一个向量的模长的平方
    qsubi = [i for i in range(emb_dim)]
    qsubj = [i for i in range(emb_dim)]
    qval = [2 for i in range(emb_dim)]
    max_vec_norm = np.dot(all_embedding[0, :], all_embedding[0, :]) + 10
    
    bound_idx_list = []
    bound_val_list = []
    bound_A_list = []

    # 下面的考虑所有已知向量同当前向量间的内积诱导的线性约束
    for known_word_idx in range(current_word_idx):  
        print(f'开始搜索第{known_word_idx+1}个单词诱导的不等式；共{current_word_idx}个单词需要考虑。', end='\r')
        # known_word_idx < current_word_idx, 所以known_word_idx作为row_i，current_word_idx作为col_j
        # 查询元素sym_mat[sym_mat_w_id, sym_mat_w_j]对应的上下界大小
        low_bound_idx, up_bound_idx = get_linear_bound_idx(ele_order_list, ij2order_id_dict, 
                                                           current_word_idx, known_word_idx, all_embedding.shape[0])
        
        bound_idx_list.append([low_bound_idx, [known_word_idx, current_word_idx], up_bound_idx])
        
        # 根据sym_mat_w_j与low_bound_idx, up_bound_idx间的关系，计算下界，
        # 下界系数，上界与上界系数（有概率下界系数与上界系数不一样）
        lb_val, lb_A, ub_val, ub_A = get_linearBound(low_bound_idx, up_bound_idx, 
                                                     current_word_idx, known_word_idx,
                                                     ij2order_id_dict, all_embedding, ele_order_list)
        
        bound_val_list.append([lb_val, ub_val])
        bound_A_list.append([lb_A, ub_A])

        if len(lb_val) == 1:
            tmp_asub = []
            tmp_aval = []
            for lb_A_i, lb_A_i_val in enumerate(lb_A):
                if lb_A_i_val != 0:
                    tmp_asub.append(lb_A_i)
                    tmp_aval.append(lb_A_i_val)
            if len(tmp_asub) >= 1:
                asub.append(tmp_asub)
                aval.append(tmp_aval)
                bkc.append(mosek.boundkey.lo)
                blc.append(lb_val[0])
                buc.append(inf)
                
        if len(ub_val) == 1:
            tmp_asub = []
            tmp_aval = []
            for ub_A_i, ub_A_i_val in enumerate(ub_A):
                if ub_A_i_val != 0:
                    tmp_asub.append(ub_A_i)
                    tmp_aval.append(ub_A_i_val)
            if len(tmp_asub) >= 1:
                asub.append(tmp_asub)
                aval.append(tmp_aval)
                bkc.append(mosek.boundkey.up)
                blc.append(-inf)
                buc.append(ub_val[0])
    return bkc, blc, buc, asub, aval, qsubi, qsubj, qval, max_vec_norm, bound_idx_list, bound_val_list, bound_A_list


def solve_min_quad(bkc, blc, buc, asub, aval, qsubi, qsubj, qval, emb_dim, if_quite=False):
    if if_quite:
        log_type = mosek.streamtype.err
    else:
        log_type = mosek.streamtype.log
    
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(log_type, streamprinter)
        # Create a task
        with env.Task(0, 0) as task:
            task.set_Stream(log_type, streamprinter)
            
            # 目标函数是最大化x1+x2+...+xn if np.random.uniform(0, 1) > 0.5 else 0 
            c = [-np.random.uniform(0, 1) for i in range(emb_dim)] 
            # 开始放入约束,先放入自变量的个数
            numvar = emb_dim
            task.appendvars(numvar)
            for i in range(numvar):  # 变量不受限制
                task.putcj(i, c[i])
                task.putvarbound(i, mosek.boundkey.fr, -inf, inf)
            task.putqobj(qsubi, qsubj, qval)
            task.putobjsense(mosek.objsense.minimize) # Input the objective sense (minimize/maximize)
            # 检查约束个数是否正确
            numcon = (len(bkc))
            task.appendcons(numcon)
            # 再放入线性约束
            print('开始放入线性约束')
            for i in tqdm(range(len(bkc))):
                con_idx = i
                task.putconbound(con_idx, bkc[i], blc[i], buc[i])
                task.putarow(con_idx,                  # Constraint (row) index.
                             asub[i],            # Column index of non-zeros in constraint i.
                             aval[i])            # Non-zero values of row i.
            # Solve the problem and print summary
            task.optimize()
            task.putintparam(mosek.iparam.intpnt_max_iterations, 400)
            task.solutionsummary(mosek.streamtype.msg)
            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr, xx)
            
            if_pdb = False
            if solsta != mosek.solsta.optimal:
                if_pdb = True
    return xx, solsta


def solve_max_linear(bkc, blc, buc, asub, aval, qsubi, qsubj, qval, emb_dim, max_vec_norm):
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)
        # Create a task
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)
            # 目标函数是最大化x1+x2+...+xn
            c = [np.random.uniform(0, 0.5) for i in range(emb_dim)]
            # 开始放入约束,先放入自变量的个数
            numvar = emb_dim
            task.appendvars(numvar)
            for i in range(numvar):  # 变量不受限制
                task.putcj(i, c[i])
                task.putvarbound(i, mosek.boundkey.fr, -inf, inf)
            task.putobjsense(mosek.objsense.maximize) # Input the objective sense (minimize/maximize)
            # 检查约束个数是否正确
            numcon = (1+len(bkc))
            task.appendcons(numcon)
            # 先放入模长最大值约束
            task.putconbound(0, mosek.boundkey.up, -inf, max_vec_norm)
            task.putqconk(0, qsubi, qsubj, qval)
            # 再放入线性约束
            print('开始放入线性约束')
            for i in tqdm(range(len(bkc))):
                con_idx = i + 1
                task.putconbound(con_idx, bkc[i], blc[i], buc[i])
                task.putarow(con_idx,                  # Constraint (row) index.
                             asub[i],            # Column index of non-zeros in constraint i.
                             aval[i])            # Non-zero values of row i.
            # Solve the problem and print summary
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr, xx)
            
            if_pdb = False
            if solsta != mosek.solsta.optimal:
                if_pdb = True
    return xx


class Node:
    def __init__(self, cord:list):
        self.cord = cord
        self.if_variable = None
        self.value = None
        self.lb_node = None
        self.ub_node = None


def build_link_list(bound_idx_list, all_embedding):
    bound_idx_node_dict = {}
    for triple_idx in bound_idx_list:  
        
        [lb_row, lb_col] = triple_idx[0]
        [known_word, current_word] = triple_idx[1]
        [ub_row, ub_col] = triple_idx[2]
        
        # 取出或创建vari_node
        vari_key = '.'.join([str(known_word), str(current_word)])
        if vari_key not in bound_idx_node_dict:
            # 需要创建对应的新的node
            vari_node = Node([known_word, current_word])
            vari_node.if_variable = True
            # 将新创建的节点存入字典
            bound_idx_node_dict[vari_key] = vari_node
        else:
            # 取出对应的node
            vari_node = bound_idx_node_dict[vari_key]
        
        # 先处理triple_idx[0] < triple_idx[1]对应的链表操作
        if lb_row is not None: 
            # 取出或创建lb_node
            lb_node_key = '.'.join([str(lb_row), str(lb_col)])
            if lb_node_key not in bound_idx_node_dict:
                # 需要创建对应的新node
                lb_node = Node([lb_row, lb_col])
                # 如果lb_col和current_word相同，表明该下界同样是一个待求的变量
                if lb_col == current_word:
                    lb_node.if_variable = True
                else:
                    lb_node.if_variable = False # 只要不是变量节点就应当可以计算出初始的数值
                    lb_node.value = np.dot(all_embedding[lb_row, :], all_embedding[lb_col, :])
                # 将新创建的节点存入字典
                bound_idx_node_dict[lb_node_key] = lb_node
            else:
                # 取出对应的node
                lb_node = bound_idx_node_dict[lb_node_key]
                
            # 填充lb_node的上界指针，vari_node的下界指针
            lb_node.ub_node = vari_node
            vari_node.lb_node = lb_node
            
        # 再处理triple_idx[1] < triple_idx[2]对应的链表操作
        if ub_row is not None: 
            # 取出或创建ub_node
            ub_node_key = '.'.join([str(ub_row), str(ub_col)])
            if ub_node_key not in bound_idx_node_dict:
                # 需要创建对应的新node
                ub_node = Node([ub_row, ub_col])
                # 如果ub_col和current_word相同，表明该下界同样是一个待求的变量
                if ub_col == current_word:
                    ub_node.if_variable = True
                else:
                    ub_node.if_variable = False # 只要不是变量节点就应当可以计算出初始的数值
                    ub_node.value = np.dot(all_embedding[ub_row, :], all_embedding[ub_col, :])
                # 将新创建的节点存入字典
                bound_idx_node_dict[ub_node_key] = ub_node
            else:
                # 取出对应的node
                ub_node = bound_idx_node_dict[ub_node_key]
                
            # 填充ub_node的下界指针，vari_node的上界指针
            ub_node.lb_node = vari_node
            vari_node.ub_node = ub_node
            
    return bound_idx_node_dict


def get_linklist_head(bound_idx_node_dict):
    return 1


def keep_low_up_bound_consist(bkc, blc, buc, asub, aval, bound_idx_list, bound_val_list, bound_A_list, ele_order_list, all_embedding):
    # 建立链表，每个节点都存储在bound_idx_node_dict里，节点间的大小关系存储在节点内的属性内。
    bound_idx_node_dict = build_link_list(bound_idx_list, all_embedding)

    pdb.set_trace()
    
    new_bkc = []
    new_blc = []
    new_buc = []
    new_asub = []
    new_aval = []
    new_bound_idx_list = []
    new_bound_val_list = []
    for i, item in enumerate(bound_idx_list):
        lb_idx = item[0]
        [known_word, current_word] = item[1]
        ub_idx = item[2]
        if current_word in lb_idx or current_word in ub_idx:  # 这个条件是半边界的条件
            
            [lb_row, lb_col] = lb_idx
            [ub_row, ub_col] = ub_idx
            
            if lb_row is not None:
                new_bkc.append(mosek.boundkey.lo)
                new_blc.append(bound_val_list[i][0][0])
                new_buc.append(inf)
                tmp_sub = []
                tmp_val = []
                for j, x in enumerate(bound_A_list[i][0]):
                    if x != 0:
                        tmp_sub.append(j)
                        tmp_val.append(x)
                if len(tmp_val) == 0:
                        pdb.set_trace()
                new_asub.append(tmp_sub)
                new_aval.append(tmp_val)
                
            if ub_row is not None:
                new_bkc.append(mosek.boundkey.up)
                new_blc.append(-inf)
                new_buc.append(bound_val_list[i][1][0])
#                 new_aval.append(bound_A_list[i][0])
                tmp_sub = []
                tmp_val = []
                for j, x in enumerate(bound_A_list[i][1]):
                    if x != 0:
                        tmp_sub.append(j)
                        tmp_val.append(x)
                if len(tmp_val) == 0:
                        pdb.set_trace()
                new_asub.append(tmp_sub)
                new_aval.append(tmp_val)
                
            new_bound_idx_list.append(item)
            new_bound_val_list.append(bound_val_list[i])
            
    return new_bkc, new_blc, new_buc, new_asub, new_aval, new_bound_idx_list, new_bound_val_list


def only_parallel_bound(bkc, blc, buc, asub, aval, bound_idx_list, bound_val_list, bound_A_list):
    new_bkc = []
    new_blc = []
    new_buc = []
    new_asub = []
    new_aval = []
    new_bound_idx_list = []
    new_bound_val_list = []
    for i, item in enumerate(bound_idx_list):
        lb_idx = item[0]
        [known_word, current_word] = item[1]
        ub_idx = item[2]
        if current_word not in lb_idx and current_word not in ub_idx:  # 这个条件是平行界的条件
            
            [lb_row, lb_col] = lb_idx
            [ub_row, ub_col] = ub_idx
            
            if len(bound_val_list[i][0]) != 0 and len(bound_val_list[i][1]) != 0:
                if bound_val_list[i][0][0] >= bound_val_list[i][1][0] or abs(bound_val_list[i][1][0] - bound_val_list[i][0][0]) < 1e-5:
                    fx_val = (bound_val_list[i][0][0] + bound_val_list[i][1][0]) / 2
                    new_bkc.append(mosek.boundkey.fx)
                    new_blc.append(fx_val)
                    new_buc.append(fx_val)
                    tmp_sub = []
                    tmp_val = []
                    for j, x in enumerate(bound_A_list[i][0]):
                        if x != 0:
                            tmp_sub.append(j)
                            tmp_val.append(x)
                    if len(tmp_val) == 0:
                        pdb.set_trace()
                    new_asub.append(tmp_sub)
                    new_aval.append(tmp_val)
                    
                    new_bound_idx_list.append(item)
                    new_bound_val_list.append([[fx_val], [fx_val]])
                    
                    continue
            
            if lb_row is not None:
                new_bkc.append(mosek.boundkey.lo)
                new_blc.append(bound_val_list[i][0][0])
                new_buc.append(inf)
                tmp_sub = []
                tmp_val = []
                for j, x in enumerate(bound_A_list[i][0]):
                    if x != 0:
                        tmp_sub.append(j)
                        tmp_val.append(x)
                if len(tmp_val) == 0:
                        pdb.set_trace()
                new_asub.append(tmp_sub)
                new_aval.append(tmp_val)
                
            if ub_row is not None:
                new_bkc.append(mosek.boundkey.up)
                new_blc.append(-inf)
                new_buc.append(bound_val_list[i][1][0])
                tmp_sub = []
                tmp_val = []
                for j, x in enumerate(bound_A_list[i][1]):
                    if x != 0:
                        tmp_sub.append(j)
                        tmp_val.append(x)
                if len(tmp_val) == 0:
                        pdb.set_trace()
                new_asub.append(tmp_sub)
                new_aval.append(tmp_val)
                
            new_bound_idx_list.append(item)
            new_bound_val_list.append(bound_val_list[i])
    return new_bkc, new_blc, new_buc, new_asub, new_aval, new_bound_idx_list, new_bound_val_list


def check_only_Bound_feas(new_bkc, new_blc, new_buc, new_asub, new_aval, new_bound_idx_list, emb_dim):
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.err, streamprinter)
        # Create a task
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.err, streamprinter)
            
            numvar = emb_dim
            task.appendvars(numvar)
            for i in range(numvar):  # 变量不受限制
                task.putvarbound(i, mosek.boundkey.fr, -inf, inf)
            task.putobjsense(mosek.objsense.minimize) # Input the objective sense (minimize/maximize)
            # 检查约束个数是否正确
            numcon = (len(new_bkc))
            task.appendcons(numcon)
            # 再放入线性约束
            for i in (range(len(new_bkc))):
                con_idx = i
                task.putconbound(con_idx, new_bkc[i], new_blc[i], new_buc[i])
                task.putarow(con_idx,                  # Constraint (row) index.
                             new_asub[i],            # Column index of non-zeros in constraint i.
                             new_aval[i])            # Non-zero values of row i.
            # Solve the problem and print summary
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)
            
            if_infeasible = False
            if solsta != mosek.solsta.optimal:
                if_infeasible = True
            
    if if_infeasible:
        np_new_aval = np.zeros([len(new_aval), emb_dim])
        for i in range(len(new_aval)):
            np_new_aval[i, 0:len(new_aval[i])] = np.array(new_aval[i])
        print(f'共{len(new_bound_idx_list)}个向量；系数矩阵的秩为：{np.linalg.matrix_rank(np_new_aval)}.')
    return if_infeasible


def check_feasibility(bkc, blc, buc, asub, aval, ele_order_list, bound_idx_list, bound_val_list, bound_A_list, emb_dim, all_embedding):
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.err, streamprinter)
        # Create a task
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.err, streamprinter)
            
            numvar = emb_dim
            task.appendvars(numvar)
            for i in range(numvar):  # 变量不受限制
                task.putvarbound(i, mosek.boundkey.fr, -inf, inf)
            task.putobjsense(mosek.objsense.minimize) # Input the objective sense (minimize/maximize)
            # 检查约束个数是否正确
            numcon = (len(bkc))
            task.appendcons(numcon)
            # 再放入线性约束
            for i in (range(len(bkc))):
                con_idx = i
                task.putconbound(con_idx, bkc[i], blc[i], buc[i])
                task.putarow(con_idx,                  # Constraint (row) index.
                             asub[i],            # Column index of non-zeros in constraint i.
                             aval[i])            # Non-zero values of row i.
            # Solve the problem and print summary
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)
            
            if_infeasible = False
            if solsta != mosek.solsta.optimal:
                if_infeasible = True
            
            if if_infeasible:
                keep_low_up_bound_consist(bkc, blc, buc, asub, aval, bound_idx_list, bound_val_list, bound_A_list, ele_order_list, all_embedding)
                
                new_bkc, new_blc, new_buc, new_asub, new_aval, \
                new_bound_idx_list, new_bound_val_list = only_parallel_bound(bkc, blc, buc, asub, aval, bound_idx_list, bound_val_list, bound_A_list)
                only_paraBound_infeasible = check_only_Bound_feas(new_bkc, new_blc, new_buc, new_asub, new_aval, new_bound_idx_list, emb_dim)
                print(f'平行界本身不可行:{only_paraBound_infeasible}')
                
                new_bkc_hp, new_blc_hp, new_buc_hp, new_asub_hp, new_aval_hp, \
                new_bound_idx_list_hp, new_bound_val_list_hp = only_half_plane_bound(bkc, blc, buc, asub, aval, bound_idx_list, bound_val_list, bound_A_list)
                only_paraBound_infeasible = check_only_Bound_feas(new_bkc_hp, new_blc_hp, new_buc_hp, new_asub_hp, new_aval_hp, new_bound_idx_list_hp, emb_dim)
                print(f'半平面界本身不可行:{only_paraBound_infeasible}')
                if only_paraBound_infeasible:
                    pdb.set_trace()
                
                new_bkc.extend(new_bkc_hp)
                new_blc.extend(new_blc_hp)
                new_buc.extend(new_buc_hp)
                new_asub.extend(new_asub_hp)
                new_aval.extend(new_aval_hp)
                new_bound_idx_list.extend(new_bound_idx_list_hp)
                new_bound_val_list.extend(new_bound_val_list_hp)
                only_paraBound_infeasible = check_only_Bound_feas(new_bkc, new_blc, new_buc, new_asub, new_aval, new_bound_idx_list, emb_dim)
                print(f'半平面界+平行界不可行:{only_paraBound_infeasible}')
                
    if if_infeasible:
        bkc = new_bkc
        blc = new_blc
        buc = new_buc
        asub = new_asub
        aval = new_aval
            
    return bkc, blc, buc, asub, aval, if_infeasible


def greedy_embed(sym_mat, all_embedding, start_emb_num, vocab_size):
    [_, emb_dim] = all_embedding.shape
    # 把sym_mat中的元素按照从小到大的顺序排列[row_i, col_j, ele_val], 其中row_i <= col_j
    ele_order_list, ij2order_id_dict = get_ele_order(sym_mat)
    # 逐个计算剩余单词的词嵌入的约束的数学描述
    for current_word_idx in range(start_emb_num, vocab_size):
        assert all_embedding[current_word_idx,:].sum() == 0
        print(f'前{all_embedding[0:current_word_idx,:].shape[0]}个向量的rank是{np.linalg.matrix_rank(all_embedding[0:current_word_idx,:])}。')
        print(f'-----------------------开始求第{current_word_idx+1}个单词的词嵌入；共{vocab_size}个单词。-----------------------')
        
        # 拿到问题描述
        bkc, blc, buc, asub, aval, \
                qsubi, qsubj, qval, max_vec_norm, bound_idx_list, bound_val_list, bound_A_list = get_pb(emb_dim, all_embedding, current_word_idx, ele_order_list, ij2order_id_dict)

        bkc, blc, buc, asub, aval, if_infeasible = check_feasibility(bkc, blc, buc, asub, aval, ele_order_list, bound_idx_list, bound_val_list, bound_A_list, emb_dim, all_embedding)
        # 总是检查求得的新向量是否总是不能用过去的向量线性表示
        not_linear_independent = True
        solsta = mosek.solsta.prim_infeas_cer
        repeat_count = 1
        while not_linear_independent or solsta == mosek.solsta.prim_infeas_cer:
            if repeat_count >= 2:
                if_quite = True
            else:
                if_quite = False
                
            min_quad_vec, solsta = solve_min_quad(bkc, blc, buc, asub, aval, qsubi, qsubj, qval, emb_dim, if_quite)
            
            # 检查线性相关性
            for i, x in enumerate(min_quad_vec):
                all_embedding[current_word_idx,i] = x
            if all_embedding[0:current_word_idx,:].shape[0] == np.linalg.matrix_rank(all_embedding[0:current_word_idx,:]):
                not_linear_independent = False
            else:
                print(f'求到了一个线性相关的向量')
                if current_word_idx+1 == vocab_size: # 最后一个向量可以线性相关
                    not_linear_independent = False
                
            if repeat_count >= 2:
                print(f'当前第{repeat_count}次求解第{current_word_idx+1}个单词的嵌入。', end='\r')
            repeat_count += 1
            
            if repeat_count >= 10:
                pdb.set_trace()
                print('重复太多次啦')
                
        vec_norm = (np.dot(all_embedding[current_word_idx,:],all_embedding[current_word_idx,:]))**0.5
        print(f'第{current_word_idx+1}个单词的词嵌入的模长为：{vec_norm}。')   
            
    return all_embedding, ele_order_list, ij2order_id_dict


if __name__ == "__main__":
    vocab_size = 10
    embedding_size = 5  # 经过尝试，580是第一个与剩下单词有非零的词义相似性
    sym_mat = generate_sym_mat(vocab_size, vocab_size)
    sym_ele_order_list, sym_ij2order_id_dict = get_ele_order(sym_mat)
    
    # 选取频率最低的embedding_size个单词，作为起始embedding.
    sub_wid_list = list(range(embedding_size))
    sub_mat = sym_mat[sub_wid_list, :][:, sub_wid_list]
    sub_ele_order_list, sub_ij2order_id_dict = get_ele_order(sub_mat)
    start_embedding = mosek_solver(sub_ele_order_list, embedding_size)
    
    aodi_embedding = np.zeros([vocab_size, vocab_size])  # 这个就是最终的embedding
    for s_w_id in sub_wid_list[0:embedding_size]:
        aodi_embedding[s_w_id, 0:embedding_size] = start_embedding[s_w_id, :]

    # 从embedding_size-1开始，向最高词频单词，逐个计算
    aodi_embedding, _, _ = greedy_embed(sym_mat, aodi_embedding, embedding_size, vocab_size)
    np.save('./toy_all_embedding.npy', aodi_embedding)
    reconstructed_pos_matrix = np.matmul(aodi_embedding, aodi_embedding.T)
    pos_ele_order_list, pos_ij2order_id_dict = get_ele_order(reconstructed_pos_matrix)
    
    U,sigma,VT = np.linalg.svd(sym_mat)
    svd_embdding = np.matmul(U, np.diag(sigma)**(0.5))
    svd_matrix = np.matmul(svd_embdding, svd_embdding.T)
    svd_ele_order_list, svd_ij2order_id_dict = get_ele_order(svd_matrix)
    
    pos_predicts = []
    svd_predicts = []
    all_expected = []
    
    for row in range(sym_ele_order_list.shape[0]-vocab_size):
        row_i = int(sym_ele_order_list[row, 0])
        col_j = int(sym_ele_order_list[row, 1])
        key = '.'.join([str(row_i), str(col_j)])
        sym_val = sym_ele_order_list[sym_ij2order_id_dict[key], 2]
        pos_val = pos_ele_order_list[pos_ij2order_id_dict[key], 2]
        svd_val = svd_ele_order_list[svd_ij2order_id_dict[key], 2]
        all_expected.append(sym_val)
        pos_predicts.append(pos_val)
        svd_predicts.append(svd_val)
    pos_pearsonr_res  = pearsonr(pos_predicts, all_expected)[0]
    pos_spearmanr_res  = spearmanr(pos_predicts, all_expected)[0]
    pos_kendalltau_res  = kendalltau(pos_predicts, all_expected)[0]
    print(pos_predicts)
    print(f'all_expected/pos_predicts的Pearson:{pos_pearsonr_res}/Spearson:{pos_spearmanr_res}/Kendalltau:{pos_kendalltau_res}.')
    svd_pearsonr_res  = pearsonr(svd_predicts, all_expected)[0]
    svd_spearmanr_res  = spearmanr(svd_predicts, all_expected)[0]
    svd_kendalltau_res  = kendalltau(svd_predicts, all_expected)[0]
    print(f'all_expected/svd_predicts的Pearson:{svd_pearsonr_res}/Spearson:{svd_spearmanr_res}/Kendalltau:{svd_kendalltau_res}.')

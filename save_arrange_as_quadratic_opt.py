# -*- coding: utf-8 -*-
import sys
import pdb
import mosek
import random
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count
# from main import load_sym_mat


inf = 0.0


def generate_sym_mat(dim: int):
    diag_ele_list = [dim + (dim**0.1)*random.uniform(1, dim) for i in range(dim)]
    sym_mat = np.zeros([dim, dim])
    for row_i in range(dim):
        for col_j in range(row_i, dim):
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


def get_problem_description(ele_order_matrix):
    [ele_num, _] = ele_order_matrix.shape
    # 目标函数为空.
    barci = []
    barcj = []
    barcval = []
    # 写入松弛变量（共ele_num个）
#     print('生成松弛变量的约束')
    blx = []
    bux = []
    bkx = []
    asub = []
    aval = []
#     for xi_id in tqdm(range(ele_num)):
    for xi_id in (range(ele_num)):
        bkx.append(mosek.boundkey.lo)
        blx.append(0)
        bux.append(+inf)
        asub.append([xi_id])
        aval.append([-1])
    # number of constraints: 原始变量 + 松弛变量
    # numcon = ele_num + ele_num
#     print('生成元素间大小的约束')
    # Bound keys for constraints
    bkc = []
    blc = []
    buc = []
    # 开始写入系数约束
    barai = []
    baraj = []
    baraval = []
    for con_id in (range(ele_num)):
#     for con_id in tqdm(range(ele_num)):
        bkc.append(mosek.boundkey.fx)
        
        if con_id < ele_num-1:  # 最大元素需要区别对待
            # Bound values for constraints
            blc.append(0)
            buc.append(0)
            # 这里假设ele_order_matrix存储的都是原矩阵的上半角元素，即ele_row <= ele_col
            [small_ele_row, small_ele_col, small_ele_val] = ele_order_matrix[con_id, :]
            [biger_ele_row, biger_ele_col, biger_ele_val] = ele_order_matrix[con_id+1, :]
            # 因为mosek都是处理下半角元素，所以需要对调一下row和col,经典的元素互换s.t. ele_row >= ele_col
            tmp_small = small_ele_row
            small_ele_row = small_ele_col
            small_ele_col = tmp_small
            tmp_biger = biger_ele_row
            biger_ele_row = biger_ele_col
            biger_ele_col = tmp_biger
            # 写入稀疏坐标
            tmp_barai = [int(biger_ele_row), int(small_ele_row)]
            tmp_baraj = [int(biger_ele_col), int(small_ele_col)]
            tmp_baraval = []
            if biger_ele_row == biger_ele_col:  # 先写入大元素的系数，再写入小元素的系数，同tmp_barai保持一致
                tmp_baraval.append(1)
            else:
                tmp_baraval.append(0.5)
            if small_ele_row == small_ele_col:
                tmp_baraval.append(-1)
            else:
                tmp_baraval.append(-0.5)
        else:
            # Bound values for constraints
            [small_ele_row, small_ele_col, small_ele_val] = ele_order_matrix[con_id, :]
            blc.append(-(small_ele_val+1))
            buc.append(-(small_ele_val+1))
            tmp_small = small_ele_row
            small_ele_row = small_ele_col
            small_ele_col = tmp_small
            tmp_barai = [int(small_ele_row)]
            tmp_baraj = [int(small_ele_col)]
            tmp_baraval = [-1]
        # 最终写入当前的约束
        barai.append(tmp_barai)
        baraj.append(tmp_baraj)
        baraval.append(tmp_baraval)
            
    return blx, bux, bkx, asub, aval, \
            barci, barcj, barcval, \
            bkc, blc, buc, barai, baraj, baraval


def mosek_solver(ele_order_matrix, dim):
    [ele_num, _] = ele_order_matrix.shape8
    assert ele_num == int(dim * (dim + 1) / 2)
    pos_mat = None
    # Make mosek environment
    with mosek.Env() as env:

        # Create a task object and attach log stream printer
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.err, streamprinter)
            
            blx, bux, bkx, asub, aval, \
            barci, barcj, barcval, \
            bkc, blc, buc, barai, baraj, baraval = get_problem_description(ele_order_matrix)

            numvar = len(bkx)  # 松弛变量的个数
            numcon = len(bkc)  # 元素排序约束的个数
            assert numvar == numcon
            BARVARDIM = [dim]
            
            # Append 'numvar' variables. 都是松弛变量
            task.appendvars(numvar)

            # Append 'numcon' empty constraints. 都是元素排序的约束
            task.appendcons(numcon)

            # Append matrix variables of sizes in 'BARVARDIM'.
            task.appendbarvars(BARVARDIM)
            
            # 开始放入问题描述
#             print('开始放入问题描述')
            symc = task.appendsparsesymmat(BARVARDIM[0],
                                           barci,
                                           barcj,
                                           barcval)
            task.putbarcj(0, [symc], [1])
            for i in (range(numcon)):
#             for i in tqdm(range(numcon)):
                # Set the bounds on variable j 松弛变量的正数约束
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(i, bkx[i], blx[i], bux[i])
                # Set the bounds on constraints.
                # blc[i] <= constraint_i <= buc[i] 
                task.putconbound(i, bkc[i], blc[i], buc[i])
                # 放入第i个元素排序约束
                
                syma = task.appendsparsesymmat(BARVARDIM[0],
                                               barai[i],
                                               baraj[i],
                                               baraval[i])
                task.putbaraij(i, 0, [syma], [1.0])
                
                task.putarow(i,                  # Constraint (row) index.
                             asub[i],            # Column index of non-zeros in constraint i.
                             aval[i])            # Non-zero values of row i.

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            # 设置优化器算法为IPM
            # task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
            # 不要浪费时间来找basis solution
            # task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
            # 设置并行的cpu数目
#             task.putintparam(mosek.iparam.num_threads, cpu_count()//5)

            # Solve the problem and print summary
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            if (solsta == mosek.solsta.optimal):
                lenbarvar = BARVARDIM[0] * (BARVARDIM[0] + 1) / 2
                barx = [0.] * int(lenbarvar)
                task.getbarxj(mosek.soltype.itr, 0, barx)
                # 填入矩阵
                pos_mat = np.zeros([dim, dim])
                barx_idx = 0
                for col_j in range(dim):
                    for row_i in range(col_j, dim):
                        pos_mat[row_i, col_j] = barx[barx_idx]
                        pos_mat[col_j, row_i] = pos_mat[row_i, col_j]
                        barx_idx += 1
            elif (solsta == mosek.solsta.dual_infeas_cer or
                  solsta == mosek.solsta.prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")
    # 做特征值分解
    if pos_mat is not None:
        eigen, ev = np.linalg.eig(pos_mat)
        if eigen.min() > 0:
            print("找到了最优解.")
            local_embedding = np.matmul(ev, np.diag(eigen**0.5))
        else:
            print("最小特征值小于0.")
            local_embedding = np.zeros([dim, dim])
    else:
        print("没有pos_mat")
        local_embedding = np.zeros([dim, dim])
    return local_embedding


def get_linear_bound_idx(ele_order_list, ij2order_id_dict, 
                         current_word_idx, known_word_idx,
                         non_zero_ele_order_idx_list,
                         known_min_idx):
    [ele_num, _] = ele_order_list.shape
    key = '.'.join([str(known_word_idx), str(current_word_idx)])
    order_id_of_ij = ij2order_id_dict[key]
    known_lb_idx = [None, None]
    known_ub_idx = [None, None]
    if ele_order_list[order_id_of_ij, 2] == 0:
#         known_ub_idx = known_min_idx
        pass
    else:
        split_idx = non_zero_ele_order_idx_list.index(order_id_of_ij)
        print('开始找下界')
        for i in tqdm(range(split_idx, -1, -1)):
            ele_order_idx = non_zero_ele_order_idx_list[i]
            row_i = ele_order_list[ele_order_idx, 0]
            col_j = ele_order_list[ele_order_idx, 1]
            # 根据row_i和col_j检查当该元素是不是已知的,
            # 有三种情况：
            if_known_val = False
            # row_i, col_j < current_word_idx
            if row_i < current_word_idx and col_j < current_word_idx:
                if_known_val = True
            # row_i = current_word_idx and col_j < current_word_idx
            elif row_i == current_word_idx and col_j < current_word_idx:
                if_known_val = True
            # col_j = current_word_idx and row_i < current_word_idx
            elif col_j == current_word_idx and row_i < current_word_idx:
                if_known_val = True

            # 若是已知元素，那么需要区别是上界还是下界
            if if_known_val:
                if ele_order_idx < order_id_of_ij:
                    known_lb_idx = [row_i, col_j]
                    break
                    
        print('开始找上界')
        for i in tqdm(range(split_idx+1, len(non_zero_ele_order_idx_list))):
            ele_order_idx = non_zero_ele_order_idx_list[i]
            row_i = ele_order_list[ele_order_idx, 0]
            col_j = ele_order_list[ele_order_idx, 1]
            # 根据row_i和col_j检查当该元素是不是已知的,
            # 有三种情况：
            if_known_val = False
            # row_i, col_j < current_word_idx
            if row_i < current_word_idx and col_j < current_word_idx:
                if_known_val = True
            # row_i = current_word_idx and col_j < current_word_idx
            elif row_i == current_word_idx and col_j < current_word_idx:
                if_known_val = True
            # col_j = current_word_idx and row_i < current_word_idx
            elif col_j == current_word_idx and row_i < current_word_idx:
                if_known_val = True

            # 若是已知元素，那么需要区别是上界还是下界
            if if_known_val:
                if ele_order_idx > order_id_of_ij:
                    known_ub_idx = [row_i, col_j]
                    break
#         for ele_order_idx in tqdm(non_zero_ele_order_idx_list):
#             ele_val = ele_order_list[ele_order_idx, 2]
#             if ele_val != 0: # 不为0的元素才有资格做上下界
#                 row_i = ele_order_list[ele_order_idx, 0]
#                 col_j = ele_order_list[ele_order_idx, 1]

#                 # 根据row_i和col_j检查当该元素是不是已知的,
#                 # 有三种情况：
#                 if_known_val = False
#                 # row_i, col_j < current_word_idx
#                 if row_i < current_word_idx and col_j < current_word_idx:
#                     if_known_val = True
#                 # row_i = current_word_idx and col_j < current_word_idx
#                 elif row_i == current_word_idx and col_j < current_word_idx:
#                     if_known_val = True
#                 # col_j = current_word_idx and row_i < current_word_idx
#                 elif col_j == current_word_idx and row_i < current_word_idx:
#                     if_known_val = True

#                 # 若是已知元素，那么需要区别是上界还是下界
#                 if if_known_val:
#                     if ele_order_idx < order_id_of_ij:
#                         known_lb_idx = [row_i, col_j]
#                     elif ele_order_idx > order_id_of_ij:
#                         known_ub_idx = [row_i, col_j]
#                         break  # 最大值只要一碰到就可以跳出for循环了
    # 结束循环后，known_lb_idx与known_ub_idx有如下可能：
    # 两者都为空
    # known_lb_idx为空
    # known_ub_idx为空
    return known_lb_idx, known_ub_idx


def get_linearBound_val_and_A(known_lb_idx, known_ub_idx, 
                              current_word_idx, known_word_idx,
                              ij2order_id_dict, all_embedding):
    [lb_row_i, lb_col_j] = known_lb_idx
    lb_val = []
    lb_A = []
    [ub_row_i, ub_col_j] = known_ub_idx
    ub_val = []
    ub_A = []
#     pdb.set_trace()
    if lb_row_i is not None:
        # 处理下界
        lb_row_i = int(lb_row_i)
        lb_col_j = int(lb_col_j)
        if current_word_idx not in [lb_row_i, lb_col_j]: # 下界是定值
            lb_val.append(np.dot(all_embedding[lb_row_i, :], all_embedding[lb_col_j, :]))
            lb_A = all_embedding[known_word_idx, :].tolist()
        else: # 
            another_vec_idx = None
            if current_word_idx == lb_row_i:  # current_word_idx不可能同时等于lb_row_i和lb_col_j
                another_vec_idx = lb_col_j
            else:
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
            another_vec_idx = None
            if current_word_idx == ub_row_i:  # current_word_idx不可能同时等于ub_row_i和ub_col_j
                another_vec_idx = ub_col_j
            else:
                another_vec_idx = ub_row_i
            ub_val.append(0)
            ub_A = (all_embedding[known_word_idx, :] - all_embedding[another_vec_idx, :]).tolist()
    return lb_val, lb_A, ub_val, ub_A


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

        np.save('./ele_order_list.npy', ele_order_list)

        with open('./ij2order_id_dict.pkl', 'wb') as f:
            pickle.dump(ij2order_id_dict, f)
        f.close()
    else:
        print(f'载入ele_order_list')
        ele_order_list = np.load('./ele_order_list.npy')
        
        print(f'载入ij2order_id_dict')
        with open('./ij2order_id_dict.pkl', 'rb') as f:
            ij2order_id_dict = pickle.load(f)
        f.close()
    
    return ele_order_list, ij2order_id_dict


def solve_feasible(bkc, blc, buc, asub, aval,
                   qsubi, qsubj, qval, max_vec_norm, emb_dim):
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
            # task.putqobj(qsubi, qsubj, qval) # 碰巧目标函数的系数矩阵同约束的系数矩阵一样
            task.putobjsense(mosek.objsense.minimize) # Input the objective sense (minimize/maximize)
            numcon = len(bkc)
            task.appendcons(numcon)
            # 先放入模长最大值约束
#             task.putconbound(0, mosek.boundkey.up, -inf, max_vec_norm)
#             task.putqconk(0, qsubi, qsubj, qval)
            # 再放入线性约束
            for i in tqdm(range(len(bkc))):
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

            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr, xx)
    return solsta, xx


def find_min_vecnormbound(bkc, blc, buc, asub, aval,
                          qsubi, qsubj, qval, last_vec_norm, emb_dim):
    # 先找到一个可行的max_vec_norm
    if_optimal, vec = solve_feasible(bkc, blc, buc, asub, aval,qsubi, qsubj, qval, last_vec_norm, emb_dim)
    # 生成tmp_asub, tmp_aval
    vec_norm = sum([x**2 for x in vec])**0.5
    if vec_norm == 0:  # 说明包含了原点，说明可以随机初始化一个向量
        vec = np.random.randn(emb_dim).tolist()
        vec_norm = sum([x**2 for x in vec])**0.5
    vec = [x/vec_norm for x in vec]
    tmp_asub = []
    tmp_aval = []
    for i in range(len(vec)):
        if vec[i] != 0:
            tmp_asub.append(i)
            tmp_aval.append(vec[i])
    return last_vec_norm, tmp_asub, tmp_aval


def greedy_embed(sym_mat, all_embedding, start_emb_num, vocab_size, if_load_new=False):
    [_, emb_dim] = all_embedding.shape
    # 把sym_mat中的元素按照从小到大的顺序排列[row_i, col_j, ele_val], 其中row_i <= col_j
    ele_order_list, ij2order_id_dict = get_ele_order(sym_mat, if_load_new=if_load_new)  
    # 找出ele_order_list中所有不为0的元素的ele_order_索引
    non_zero_ele_order_idx_list = np.nonzero(ele_order_list[:, 2])[0].tolist()
    # 下面的变量用来跟踪已知最小元
    known_min_idx = [start_emb_num-1, 0]
    known_min_val = sym_mat[start_emb_num-1, 0]
#     pdb.set_trace()
    # 逐个计算剩余单词的词嵌入的约束的数学描述
    for current_word_idx in range(start_emb_num, vocab_size):
        assert all_embedding[current_word_idx,:].sum() == 0
        print(f'-----------------------开始求第{current_word_idx}个单词的词嵌入；共{vocab_size}个单词。-----------------------')
        with mosek.Env() as env:
            # Attach a printer to the environment
            env.set_Stream(mosek.streamtype.log, streamprinter)
            # Create a task
            with env.Task(0, 0) as task:
                task.set_Stream(mosek.streamtype.log, streamprinter)
                
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
                last_vec_norm = np.dot(all_embedding[current_word_idx-1, :], all_embedding[current_word_idx-1, :])
                
                numcon = 1  # 上面的模长最小值约束已经算一个约束了
                # 下面的考虑所有已知向量同当前向量间的内积诱导的线性约束
                for known_word_idx in range(current_word_idx):  
                    print(f'开始搜索第{known_word_idx+1}个单词诱导的不等式；共{current_word_idx-1}个单词需要考虑。', end='\r')
                    # known_word_idx < current_word_idx, 所以known_word_idx作为row_i，current_word_idx作为col_j
                    # 查询元素sym_mat[sym_mat_w_id, sym_mat_w_j]对应的上下界大小
                    low_bound_idx, up_bound_idx = get_linear_bound_idx(ele_order_list, ij2order_id_dict, 
                                                                       current_word_idx, known_word_idx,
                                                                       non_zero_ele_order_idx_list,
                                                                       known_min_idx)

                    # 根据sym_mat_w_j与low_bound_idx, up_bound_idx间的关系，计算下界，
                    # 下界系数，上界与上界系数（有概率下界系数与上界系数不一样）
                    lb_val, lb_A, ub_val, ub_A = get_linearBound_val_and_A(low_bound_idx, up_bound_idx, 
                                                                           current_word_idx, known_word_idx,
                                                                           ij2order_id_dict, all_embedding)

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
                            numcon += 1
                            bkc.append(mosek.boundkey.lo)
                            blc.append(lb_val[0])
                            buc.append(inf)
#                         assert lb_val[0] <= ub_val[0]
                    if len(ub_val) == 1:
                        if len(lb_val) == 0 or lb_val[0] < ub_val[0]:
                            tmp_asub = []
                            tmp_aval = []
                            for ub_A_i, ub_A_i_val in enumerate(ub_A):
                                if ub_A_i_val != 0:
                                    tmp_asub.append(ub_A_i)
                                    tmp_aval.append(ub_A_i_val)
                            if len(tmp_asub) >= 1:
                                asub.append(tmp_asub)
                                aval.append(tmp_aval)
                                numcon += 1
                                bkc.append(mosek.boundkey.up)
                                blc.append(-inf)
                                buc.append(ub_val[0])
                        
                        
                    # 更新已知最小元
                    if sym_mat[known_word_idx, current_word_idx] != 0:
                        if known_min_val > sym_mat[known_word_idx, current_word_idx]:
                            known_min_val = sym_mat[known_word_idx, current_word_idx]
                            known_min_idx = [known_word_idx, current_word_idx]
                            
                        
                # 开始放入约束,先放入自变量的个数
                numvar = emb_dim
                task.appendvars(numvar)
                for i in range(numvar):  # 变量不受限制
                    task.putvarbound(i, mosek.boundkey.fr, -inf, inf)
                task.putqobj(qsubi, qsubj, qval) # 碰巧目标函数的系数矩阵同约束的系数矩阵一样
                task.putobjsense(mosek.objsense.minimize) # Input the objective sense (minimize/maximize)
                # 检查约束个数是否正确
                assert numcon == (1+len(bkc))
#                 numcon -= 1
                task.appendcons(numcon)
                # 先放入模长最小值约束,用线性约束代替concave的模长最小值约束
#                 task.putconbound(0, mosek.boundkey.lo, last_vec_norm, inf)
#                 task.putqconk(0, qsubi, qsubj, qval)
                min_vec_norm, tmp_asub, tmp_aval = find_min_vecnormbound(bkc, blc, buc, asub, aval,
                                                                         qsubi, qsubj, qval, last_vec_norm,
                                                                         emb_dim)
                task.putconbound(0, mosek.boundkey.lo, np.dot(all_embedding[0, :], all_embedding[0, :]), +inf)
                task.putarow(0, tmp_asub, tmp_aval)
                # 再放入线性约束
                print('开始放入线性约束')
                for i in tqdm(range(len(bkc))):
                    con_idx = i + 1
#                     con_idx = i
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
                # 填入嵌入
                assert len(xx) == emb_dim
                for i, x in enumerate(xx):
                    all_embedding[current_word_idx,i] = x
                print(f'第{current_word_idx}个单词的词嵌入的模长为：{(np.dot(all_embedding[current_word_idx,:],all_embedding[current_word_idx,:]))**0.5}。')

#                 if (solsta == mosek.solsta.optimal):
                    
                if (solsta == mosek.solsta.dual_infeas_cer or
                      solsta == mosek.solsta.prim_infeas_cer):
                    print("Primal or dual infeasibility certificate found.\n")
                    new_vec_norm = np.dot(all_embedding[0, :], all_embedding[0, :])
                    vec = np.random.randn(emb_dim).tolist()
                    vec_norm = sum([x**2 for x in vec])**0.5
                    vec = [x/vec_norm*new_vec_norm for x in vec]
                    for i, x in enumerate(vec):
                        all_embedding[current_word_idx,i] = x
                    
                    
    return all_embedding


def get_sym_mat(if_load_new: bool = False):
    if if_load_new is not True:
        # 载入对称矩阵
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
    
        # 填入对角线元素，sym_mat[i,i]>sym_mat[i+1,i+1]
        max_am = sym_mat.max()
        [vocab_size, _] = sym_mat.shape
        delta_length = 10 / vocab_size
        for w_id in range(len(word2id_dict)):
            sym_mat[w_id, w_id] = max_am + 10.5 - delta_length * (w_id+1)
        
        print('翻转一下对称矩阵')
        new_sym_mat = np.zeros(sym_mat.shape)
        for new_row_i in tqdm(range(vocab_size)):
            old_row_i = vocab_size - 1 - new_row_i
            for new_col_j in range(new_row_i, vocab_size):
                old_col_j = vocab_size - 1 - new_col_j
                new_sym_mat[new_row_i, new_col_j] = sym_mat[old_row_i, old_col_j]
                new_sym_mat[new_col_j, new_row_i] = new_sym_mat[new_row_i, new_col_j]

        new_word2id_dict = {}
        new_id2word_dict = {}
        for key in new_word2id_dict.keys():
            new_id = vocab_size - 1 - word2id_dict[key]
            new_word2id_dict[key] = new_id
            new_id2word_dict[new_id] = key
            
        print(f'保存new_sym_mat')
        np.save('./new_sym_mat.npy', new_sym_mat)

        print(f'保存new_word2id_dict')
        with open('./new_word2id_dict.pkl', 'wb') as f:
            pickle.dump(new_word2id_dict, f)
        f.close()

        print(f'保存new_id2word_dict')
        with open('./new_id2word_dict.pkl', 'wb') as f:
            pickle.dump(new_id2word_dict, f)
        f.close()
    else:
        print(f'载入new_sym_mat')
        new_sym_mat = np.load('./new_sym_mat.npy')

        print(f'载入new_word2id_dict')
        with open('./new_word2id_dict.pkl', 'rb') as f:
            new_word2id_dict = pickle.load(f)
        f.close()

        print(f'载入new_id2word_dict')
        with open('./new_id2word_dict.pkl', 'rb') as f:
            new_id2word_dict = pickle.load(f)
        f.close()
        
    return new_sym_mat, new_word2id_dict, new_id2word_dict


if __name__ == "__main__":
    if_load_new=True
    sym_mat, word2id_dict, id2word_dict = get_sym_mat(if_load_new=if_load_new)
    [vocab_size, _] = sym_mat.shape
    
    embedding_size = 580  # 经过尝试，580是第一个与剩下单词有非零的词义相似性
    all_embedding = np.zeros([vocab_size, embedding_size])  # 这个就是最终的embedding
    
    # 选取频率最低的embedding_size个单词，作为起始embedding.
    sub_wid_list = list(range(embedding_size))
    sub_mat = sym_mat[sub_wid_list, :][:, sub_wid_list]
    diag, col_vec = np.linalg.eig(sub_mat)
    if diag.min() > 0:  # 当前矩阵已经是正定矩阵，无需求解
        start_embedding = np.matmul(col_vec, np.diag(diag)**0.5)  # 一行是一个向量
        all_embedding[sub_wid_list, :] = start_embedding[sub_wid_list, :]
        # 从embedding_size-1开始，向最高词频单词，逐个计算
        all_embedding = greedy_embed(sym_mat, all_embedding, embedding_size, vocab_size, if_load_new=if_load_new)
        np.save('./all_embedding.npy', all_embedding)
    else:  # 否则需要在gram层面求解对应的保序正定近似
        ele_order_matrix = get_ele_order(sub_mat)
        local_embedding = mosek_solver(ele_order_matrix, vocab_size)

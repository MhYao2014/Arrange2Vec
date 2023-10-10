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


def get_ele_order(sym_mat):
    [vocab_size, _] = sym_mat.shape
    num_ele = int(vocab_size * (vocab_size + 1) / 2)
    ele_order_list = np.zeros([num_ele, 3])
    ele_idx_in_list = 0;
    for row_i in range(vocab_size):
        for col_j in range(row_i, vocab_size):
#             print(f'row:{row_i};col_j:{col_j}', end='\r')
            ele_order_list[ele_idx_in_list, 0] = row_i
            ele_order_list[ele_idx_in_list, 1] = col_j
            ele_order_list[ele_idx_in_list, 2] = sym_mat[row_i, col_j]
            ele_idx_in_list += 1
    ele_order_list = ele_order_list[ele_order_list[:,2].argsort(),:]
    return ele_order_list


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
    [ele_num, _] = ele_order_matrix.shape
    assert ele_num == int(dim * (dim + 1) / 2)
    pos_mat = None
    # Make mosek environment
    with mosek.Env() as env:

        # Create a task object and attach log stream printer
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)
            
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
            print('开始放入问题描述')
            symc = task.appendsparsesymmat(BARVARDIM[0],
                                           barci,
                                           barcj,
                                           barcval)
            task.putbarcj(0, [symc], [1])
#             for i in (range(numcon)):
            for i in tqdm(range(numcon)):
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


if __name__ == "__main__":
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
    # 填入对角线元素
    max_am = sym_mat.max()
    [dim, _] = sym_mat.shape
    delta_length = 10 / dim
    for w_id in range(len(word2id_dict)):
        sym_mat[w_id, w_id] = max_am + 10.5 - delta_length * (w_id+1)
        
    vocab_size = 100
    sub_wid_list = list(range(vocab_size))
    sub_mat = sym_mat[sub_wid_list, :][:, sub_wid_list]
    ele_order_matrix = get_ele_order(sub_mat)
    # call the main function
#     vocab_size = 100
#     sym_mat = generate_sym_mat(vocab_size)
#     ele_order_matrix = get_ele_order(sym_mat)  # [row_i, col_j, mat_val]
    
    try:
        local_embedding = mosek_solver(ele_order_matrix, vocab_size)
        np.save(f'./test_{vocab_size}_embed.npy', local_embedding)
    except mosek.MosekException as e:
        print("ERROR: %s" % str(e.errno))
        if e.msg is not None:
            print("\t%s" % e.msg)
            sys.exit(1)
    except:
        import traceback
        traceback.print_exc()
        sys.exit(1)

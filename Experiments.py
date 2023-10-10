# -*- coding: utf-8 -*-
import sys
import pdb
import mosek
import random
import pickle
import xlwt
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau
from save_arrange_as_quad_and_linear_opt import get_ele_order, generate_sym_mat
from sym2pos_save_arrange_as_sdp import mosek_solver


def save_excel(sym_expected, svd_predicts, pos_predicts, exp_id, vocab_size, pearsonr_sheet, spearmanr_sheet, kendalltau_sheet):
    
    
    # 保存该规模下的实验结果
    pos_pearsonr_res  = pearsonr(pos_predicts, sym_expected)[0]
    svd_pearsonr_res  = pearsonr(svd_predicts, sym_expected)[0]
    
    pearsonr_sheet.write(2+exp_id, 0, str(vocab_size))
    pearsonr_sheet.write(2+exp_id, 1, str(pos_pearsonr_res))
    pearsonr_sheet.write(2+exp_id, 2, str(svd_pearsonr_res))
    
    pos_spearmanr_res  = spearmanr(pos_predicts, sym_expected)[0]
    svd_spearmanr_res  = spearmanr(svd_predicts, sym_expected)[0]
    
    spearmanr_sheet.write(2+exp_id, 0, str(vocab_size))
    spearmanr_sheet.write(2+exp_id, 1, str(pos_spearmanr_res))
    spearmanr_sheet.write(2+exp_id, 2, str(svd_spearmanr_res))
    
    pos_kendalltau_res  = kendalltau(pos_predicts, sym_expected)[0]
    svd_kendalltau_res  = kendalltau(svd_predicts, sym_expected)[0]
    
    kendalltau_sheet.write(2+exp_id, 0, str(vocab_size))
    kendalltau_sheet.write(2+exp_id, 1, str(pos_kendalltau_res))
    kendalltau_sheet.write(2+exp_id, 2, str(svd_kendalltau_res))


def pearson_expriments():
    workbook=xlwt.Workbook(encoding='utf-8')
    
    pearsonr_sheet=workbook.add_sheet('Pearsonr', cell_overwrite_ok=True)
    pearsonr_sheet.write(0, 0, 'Size')
    pearsonr_sheet.write(0, 1, 'Gram (ours)')
    pearsonr_sheet.write(0, 2, 'SVD')
    
    spearmanr_sheet=workbook.add_sheet('Spearmanr', cell_overwrite_ok=True)
    spearmanr_sheet.write(0, 0, 'Size')
    spearmanr_sheet.write(0, 1, 'Gram (ours)')
    spearmanr_sheet.write(0, 2, 'SVD')
    
    kendalltau_sheet=workbook.add_sheet('Kendalltau', cell_overwrite_ok=True)
    kendalltau_sheet.write(0, 0, 'Size')
    kendalltau_sheet.write(0, 1, 'Gram (ours)')
    kendalltau_sheet.write(0, 2, 'SVD')
    
    for exp_id, vocab_size in enumerate(range(10, 100, 10)):
        print(f'-----------------------开展{vocab_size}规模的对比实验-----------------------')
        # 生成一个随机的非正定对称矩阵。
        sym_mat = generate_sym_mat(vocab_size, vocab_size)
        sym_ele_order_list, sym_ij2order_id_dict = get_ele_order(sym_mat)
        
        # 应用Gram方法求解嵌入。
        gram_embedding = mosek_solver(sym_ele_order_list, vocab_size)
        gram_pos_matrix = np.matmul(gram_embedding, gram_embedding.T)
        pos_ele_order_list, pos_ij2order_id_dict = get_ele_order(gram_pos_matrix)

        # 应用SVD方法求解嵌入。
        U,sigma,VT = np.linalg.svd(sym_mat)
        svd_embdding = np.matmul(U, np.diag(sigma)**(0.5))
        svd_matrix = np.matmul(svd_embdding, svd_embdding.T)
        svd_ele_order_list, svd_ij2order_id_dict = get_ele_order(svd_matrix)
        
        # 对齐各个方法的预测结果
        pos_predicts = []
        svd_predicts = []
        sym_expected = []

        for row in range(sym_ele_order_list.shape[0]-vocab_size):
            row_i = int(sym_ele_order_list[row, 0])
            col_j = int(sym_ele_order_list[row, 1])
            key = '.'.join([str(row_i), str(col_j)])
            sym_val = sym_ele_order_list[sym_ij2order_id_dict[key], 2]
            pos_val = pos_ele_order_list[pos_ij2order_id_dict[key], 2]
            svd_val = svd_ele_order_list[svd_ij2order_id_dict[key], 2]
            sym_expected.append(sym_val)
            pos_predicts.append(pos_val)
            svd_predicts.append(svd_val)
            
        # 保存实验结果到excel文件中去
        save_excel(sym_expected, svd_predicts, pos_predicts, exp_id, vocab_size, pearsonr_sheet, spearmanr_sheet, kendalltau_sheet)
        if vocab_size % 100 == 0: 
            workbook.save(f'pearson_expriments_vs_{vocab_size}.xls')

if __name__ == "__main__":
    # 开展pearson实验
    pearson_expriments()
    
    
    

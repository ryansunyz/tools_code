from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def load_set_data(file_name):
    data_mat = []
    label_mat = []
    n = 0

    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split(",")
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
        n += 1

    return data_mat, label_mat, n

def select_j_rand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))

    return j

def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L

    return aj

def smo_simple(data_matin, class_labele, C, toler, max_iter):
    data_matrix = mat(data_matin)  #将输入列表转成矩阵
    label_mat = mat(class_labele).transpose()  #将训练数据转成列向量
    b = 0
    m,n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while(iter < max_iter):
        alpha_pirs_change = 0
        for i in range(m):
            fxi = float(multiply(alphas, label_mat).T * (data_matrix * (data_matrix[i, :].T))) + b
            ei = fxi - float(label_mat[i])
            #if ((label_mat[i] * ei < -toler) and (alphas[i] < C)) or \
            #    ((label_mat[i] * ei > toler) and \
            #    (alphas[i] > 0)):
            if((label_mat[i] * (fxi - 2 * b) <= 1 and alphas[i] < C)\
                or (label_mat[i] * (fxi - 2 * b) >= 1 and alphas[i] > 0)\
                or (label_mat[i] * (fxi - 2 * b) == 1 and (alphas[i] == 0 or alphas[i] == C))):
                j = select_j_rand(i, m)  #随机选择aj且i != j，相当于随机选择ai和aj
                fxj = float(multiply(alphas, label_mat).T * \
                    (data_matrix * data_matrix[j, :].T)) + b
                ej = fxj - float(label_mat[j])
                alpha_iold = alphas[i].copy()  #保存alphai更新前的值
                alpha_jold = alphas[j].copy()  #保存alphaj更新前的值
                #求解alphaj的上下边界
                if (label_mat[i] != label_mat[j]):
                    L = max(0, alpha_jold - alpha_iold)
                    H = min(C, C + alpha_jold + alpha_iold)
                else:
                    L = max(0, alpha_jold + alpha_iold - C)
                    H = min(C, alpha_iold + alpha_jold)
                if (L == H):
                    print("L == H")
                    continue
                eta = 2 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T
                if (eta > 0):
                    print("eta > 0")
                    continue
                alphas[j] = alpha_jold - (label_mat[j] * (ei - ej) * 1.0 / eta)
                alphas[j] = clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alpha_jold) < 0.00001):  
                    print("j not moving enough")
                    continue
                alphas[i] = alpha_iold + (label_mat[i] * label_mat[j] * (alpha_jold - alphas[j]))
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_iold) * (data_matrix[i, :] * data_matrix[i, :].T)\
                    - label_mat[j] * (alphas[j] - alpha_jold) * (data_matrix[i, :] * data_matrix[j, :].T)
                b2 = b - ej - label_mat[i] * (alphas[i] - alpha_iold) * (data_matrix[i, :] * data_matrix[j, :].T)\
                    - label_mat[j] * (alphas[j] - alpha_jold) * (data_matrix[j, :] * data_matrix[j, :].T) 
                if (alphas[i] > 0 and alphas[i] < C):
                    b = b1
                elif (alphas[j] > 0 and alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pirs_change += 1
                print("iter: %d i: %d, paris changed % d" % (iter, i, alpha_pirs_change))
        if (alpha_pirs_change == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b,alphas

def show_experiment_plot(alphas, data_list_in, label_list_in, b, n):
    data_arr_in = array(data_list_in)
    label_arr_in = array(label_list_in)
    alphas_arr = alphas.getA()
    data_mat = mat(data_list_in)
    label_mat = mat(label_list_in).transpose()

    i = 0
    weights = zeros((2, 1))
    while(i < n):
        if(label_arr_in[i] == -1):
            plt.plot(data_arr_in[i, 0], data_arr_in[i, 1], "ob")
        elif(label_arr_in[i] == 1):
            plt.plot(data_arr_in[i, 0], data_arr_in[i, 1], "or")
        if(alphas_arr[i] > 0):
            plt.plot(data_arr_in[i, 0], data_arr_in[i, 1], "oy")
            weights += multiply(alphas[i] * label_mat[i], data_mat[i, :].T)
        i += 1

    x = arange(-2, 12, 0.1)
    y = []
    for k in x:
        y.append(float(-b - weights[0] * k) / weights[1])

    plt.plot(x, y, '-g')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def main():
    data_list,label_list, n = load_set_data("test_set.txt")
    b,alphas = smo_simple(data_list, label_list, 0.6, 0.001, 40)
    b_data = array(b)[0][0]
    show_experiment_plot(alphas, data_list, label_list, b_data, n)

main()

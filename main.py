import csv
import torch
import numpy as np
import numba as nb

################################################################################
def Read_CSV(FileName):
    #
    with open(FileName, newline = '') as csvfile:
        data = csv.reader(csvfile, dialect = 'excel')
        data_list = []
        for line in data:
            data_list.append(line)
    #
    return data_list

################################################################################
def Extract_Item_Features(list):
    #
    N = len(list)
    d = len(list[0])
    String_Types = [[] for i in range(5)]
    list_array = np.zeros((N - 1, 5))
    Targets_array = np.zeros((N - 1, d - 6))
    #
    for i in range(N - 1):
        for j in range(1, 6):
            if list[i + 1][j] in String_Types[j - 1]:
                list_array[i, j - 1] = String_Types[j - 1].index(list[i + 1][j]) + 1
            else:
                String_Types[j - 1].append(list[i + 1][j])
                list_array[i, j - 1] = len(String_Types[j - 1])
        #
        for j in range(6, d):
            Targets_array[i, j - 6] = float(list[i + 1][j])
    #
    return(String_Types, list_array, Targets_array)

################################################################################
def Extract_Date_Features(list, f_cols):
    #
    N = len(list)
    d = len(list[0])
    String_Types = [[] for i in range(len(f_cols) - 2)]
    list_array = np.zeros((N - 1, len(f_cols)))
    #
    for i in range(N - 1):
        for S_j, j in enumerate(f_cols):
            if j >= 9:
                #
                if list[i + 1][j] in String_Types[S_j - 2]:
                    list_array[i, S_j] = String_Types[S_j - 2].index(list[i + 1][j]) + 1
                else:
                    String_Types[S_j - 2].append(list[i + 1][j])
                    list_array[i, S_j] = len(String_Types[S_j - 2])
            else:
                if list[i + 1][j] in String_Types[S_j]:
                    list_array[i, S_j] = String_Types[S_j].index(list[i + 1][j]) + 1
                else:
                    String_Types[S_j].append(list[i + 1][j])
                    list_array[i, S_j] = len(String_Types[S_j])
    #
    return(String_Types, list_array)

################################################################################
def Extract_Store_Item_Date_Features(list, Store_Types, Item_Types, Date_Types):
    #
    N = len(list)
    N1 = len(Store_Types)
    N2 = len(Item_Types)
    N3 = len(Date_Types)
    list_array = np.ones((N1, N2, N3)) * 100
    #
    for i in range(N - 1):
        n1 = Store_Types.index(list[i + 1][0])
        n2 = Item_Types.index(list[i + 1][1])
        n3 = Date_Types.index(list[i + 1][2])
        list_array[n1, n2, n3] = float(list[i + 1][3])
    #
    return(list_array)

################################################################################
#@nb.jit(nopython = True, nogil = True)
def Centralization_and_Standardization(array_list, mode = 'cols'):
    if len(array_list) != 1:
        N1, d1 = array_list[0].shape
        N2, d2 = array_list[1].shape
        if d1 == d2:
            array_merge = np.concatenate((array_list[0], array_list[1]), axis = 0)
            d = d1
            N = N1 + N2
        elif N1 == N2:
            array_merge = np.concatenate((array_list[0], array_list[1]), axis = 1)
            N = N1
            d = d1 + d2
        else:
            print('Error')
    else:
        if array_list[0].ndim == 2:
            N, d = array_list[0].shape
        array_merge = array_list[0]
    #
    if mode == 'cols':
        CS = np.zeros((d, 2))
        for i in range(d):
            CS[i, 0] = np.mean(array_merge[:, i])
            array_merge[:, i] -= CS[i, 0]
            CS[i, 1] = np.std(array_merge[:, i])
            array_merge[:, i] /= CS[i, 1]
    elif mode == 'whole':
        CS = np.zeros(2)
        CS[0] = np.mean(array_merge)
        array_merge -= CS[0]
        CS[1] = np.std(array_merge)
        array_merge /= CS[1]
    #
    if len(array_list) != 1:
        if d1 == d2:
            return(array_merge[:N1, :], array_merge[N1:, :], CS)
        elif N1 == N2:
            return(array_merge[:, :d1], array_merge[:, d1:], CS)
    else:
        return(array_merge, CS)

################################################################################
def Consolidation_and_Save(evaluation_Train, evaluation_Target, validation_Train, validation_Target, calendar, sell_prices):
    #evaluation
    N1, d1 = evaluation_Train.shape
    N2, d2 = evaluation_Target.shape
    N3, d3 = calendar.shape
    N_row = N1 * d2
    N_Item_Batch = 300
    #
    Count = 0
    N_Batch = 0
    while Count < N1:
        with open('Evaluation_Part%d.csv' % (N_Batch), 'w') as out:
            for i in range(Count, N1):
                for j in range(d2):
                    print(','.join([str(evaluation_Train[i, d]) for d in range(d1)]), end = ',', file = out)
                    print(','.join([str(calendar[j, d]) for d in range(d3) if d != 0]), end = ',', file = out)
                    print(sell_prices[int(evaluation_Train[i, 3]), int(evaluation_Train[i, 0]), int(calendar[j, 0])], end = ',', file = out)
                    print(evaluation_Target[i, j], end = '\n', file = out)
                if i % N_Item_Batch == -1:
                    break
        N_Batch += 1
        Count += N_Item_Batch

################################################################################
def main():
    #
    calendar_list = Read_CSV('calendar.csv')
    sales_train_evaluation_list = Read_CSV('sales_train_evaluation.csv')
    sales_train_validation_list = Read_CSV('sales_train_validation.csv')
    sell_prices_list = Read_CSV('sell_prices.csv')
    #商品信息
    sales_train_evaluation_Types, sales_train_evaluation_array, train_evaluation_Target_array = Extract_Item_Features(sales_train_evaluation_list)
    sales_train_validation_Types, sales_train_validation_array, train_validation_Target_array = Extract_Item_Features(sales_train_validation_list)
    #日历信息，价格信息
    calendar_f_indexs = [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13]
    calendar_Types, calendar_array = Extract_Date_Features(calendar_list, calendar_f_indexs)
    if sales_train_evaluation_Types == sales_train_validation_Types:
        print('Types in evaluation and validation are the same!')
    else:
        print('Error')
    sell_prices_array = Extract_Store_Item_Date_Features(sell_prices_list, sales_train_evaluation_Types[3], sales_train_evaluation_Types[0], calendar_Types[0])
    #中心化与规范化
    #np.set_printoptions(threshold = np.inf)
    sales_train_evaluation_array, sales_train_validation_array, sales_train_CS = Centralization_and_Standardization([sales_train_evaluation_array, sales_train_validation_array])
    calendar_array, calendar_CS = Centralization_and_Standardization([calendar_array])
    sell_prices_array, sell_prices_CS = Centralization_and_Standardization([sell_prices_array], 'whole')
    train_evaluation_Target_array, train_validation_Target_array, train_Target_CS = Centralization_and_Standardization([train_evaluation_Target_array, train_validation_Target_array], 'whole')
    #整合数据保存
    Consolidation_and_Save(sales_train_evaluation_array, train_evaluation_Target_array, sales_train_validation_array, train_validation_Target_array, calendar_array, sell_prices_array)

################################################################################
if __name__ == '__main__':
    main()

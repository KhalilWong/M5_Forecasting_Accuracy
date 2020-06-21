import csv
import torch
import numpy as np

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
def Extract_Features(list):
    #
    N = len(list)
    d = len(list[0])
    String_Types = [[] for i in range(6)]
    list_array = np.zeros((N, 6))
    #
    for i in range(N):
        for j in range(6):
            if list[i][j] in String_Types[j]:
                list_array[i, j] = String_Types[j].index(list[i][j])
            else:
                String_Types[j].append(list[i][j])
                list_array[i, j] = len(String_Types[j]) - 1
    #
    return(String_Types, list_array)

################################################################################
def main():
    #
    calendar_list = Read_CSV('calendar.csv')
    sales_train_evaluation_list = Read_CSV('sales_train_evaluation.csv')
    sales_train_validation_list = Read_CSV('sales_train_validation.csv')
    sell_prices_list = Read_CSV('sell_prices.csv')
    #
    sales_train_evaluation_Types, sales_train_evaluation_array = Extract_Features(sales_train_evaluation_list)
    print(sales_train_evaluation_array[:, 0])
    print(sales_train_evaluation_array[:, 5])

################################################################################
if __name__ == '__main__':
    main()

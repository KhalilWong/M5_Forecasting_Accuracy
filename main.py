import csv
import torch

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
def main():
    calendar_list = Read_CSV('calendar.csv')
    sales_train_evaluation_list = Read_CSV('sales_train_evaluation.csv')
    sales_train_validation_list = Read_CSV('sales_train_validation.csv')
    sell_prices_list = Read_CSV('sell_prices.csv')

################################################################################
if __name__ == '__main__':
    main()

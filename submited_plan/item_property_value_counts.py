#coding:utf8
import pandas as pd
import warnings
import re

if __name__ == "__main__":
    #忽略警告
    warnings.filterwarnings("ignore")
    data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    # print(data.item_id.value_counts())
    # 此处显示item_id最多出现次数为3000次，最少1次
    # 一共由10075种不同的item_id
    prop_lst = data.item_property_list
    length = len(prop_lst)
    tmp = []
    for i in range(length):
        try:
            tmp += re.split(';',prop_lst[i])
            # print(i)
        except:
            continue
    tmp = pd.Series(tmp)
    print(tmp.value_counts())
    # 此处显示item_property_list最多出现次数为413357次，最少1次
    # 一共由61399种不同的item_property_list
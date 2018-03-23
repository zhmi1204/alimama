#coding:utf8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 当用行号索引的时候, 尽量用 iloc 来进行索引;
# 而用标签索引的时候用 loc
# 尽量不用ix（因为ix既对行，也可对列）
#装载数据
data_csv = pd.read_table('round1_ijcai_18_train_20180301.txt',sep=' ')
data_origin = pd.DataFrame(data_csv)

##################################其他测试操作#####################################
#第一行的字段属性
# print(data_origin[0:0])
# print("所有数据打印： ",data_origin) 478138条
# print(len(data_origin))

# 按照某些字段去重，后面所有对源数据操作，以data_origin为准
data_origin.drop_duplicates(['instance_id', 'item_id', 'item_category_list', 'item_property_list', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id', 'context_timestamp', 'context_page_id', 'predict_category_property', 'shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'is_trade'])
# print(len(data_origin))

#####################################is_trade操作######################################
#字段is_trade的0和1的记录
# no_buy = data_origin[data_origin['is_trade']==0]
# is_buy = data_origin[data_origin['is_trade']==1]

#获取字段is_grade=0或1时的所有记录
# print(data_origin[data_origin['is_trade']==0])

#获取字段is_grade=0或1时记录的索引index
# print(data_origin[data_origin['is_trade']==1].index)

# 字段is_trade内不同分类的分布（按出现数量）某记录出现的频率,返回值Series
# print(data_origin['is_trade'].value_counts())

#获得is_trade0和is_trade1的索引index，返回值list
index_istrade0_list = data_origin[data_origin['is_trade']==0].index.tolist()
index_istrade1_list = data_origin[data_origin['is_trade']==1].index.tolist()

#获得is_trade的记录的索引
# print("买了： ",type(index_istrade0_list))
# print("没买： ",type(index_istrade1_list))

####################################user_gender_id操作#################################
#查看user_gender_id四个不同属性出现的频次
# 0    360817
# 1     94070
# -1    12902
# 2     10349
# print(data_origin['user_gender_id'].value_counts())

#获得(user_gender_id)&(is_trade)，各gender用户买或没买的记录的索引index，返回值list
usergender0_list_trade1 = data_origin[(data_origin['user_gender_id']==0)&(data_origin['is_trade']==1)].index.tolist()
usergender1_list_trade1 = data_origin[(data_origin['user_gender_id']==1)&(data_origin['is_trade']==1)].index.tolist()
usergender_1_list_trade1 = data_origin[(data_origin['user_gender_id']==-1)&(data_origin['is_trade']==1)].index.tolist()
usergender2_list_trade1 = data_origin[(data_origin['user_gender_id']==2)&(data_origin['is_trade']==1)].index.tolist()

usergender0_list_trade0 = data_origin[(data_origin['user_gender_id']==0)&(data_origin['is_trade']==0)].index.tolist()
usergender1_list_trade0 = data_origin[(data_origin['user_gender_id']==1)&(data_origin['is_trade']==0)].index.tolist()
usergender_1_list_trade0 = data_origin[(data_origin['user_gender_id']==-1)&(data_origin['is_trade']==0)].index.tolist()
usergender2_list_trade0 = data_origin[(data_origin['user_gender_id']==2)&(data_origin['is_trade']==0)].index.tolist()

# print(usergender0_list_trade0)
#########################################数据类型不太对，或者其他地方不对#########################################
#绘制两个数据堆叠的直方图
#准备画布
plt.figure()
#
#准备plt.bar数据，横轴数据，纵轴数据，label，放在底部的bottom
user_gender = [0,1,-1,2]
isbuy = [len(usergender0_list_trade1) , len(usergender1_list_trade1) , len(usergender_1_list_trade1) , len(usergender2_list_trade1)]
nobuy = [len(usergender0_list_trade0) , len(usergender1_list_trade0) , len(usergender_1_list_trade0) , len(usergender2_list_trade0)]
#
#画直方图
plt.bar(user_gender, isbuy, label='isbuy')
plt.bar(user_gender, nobuy, bottom=isbuy, label='nobuy')
#
#坐标轴使用刻度
plt.grid(True)
#覆盖plt.bar里面的横轴
plt.xticks(user_gender)
#显示图例
plt.legend(loc=0)
#显示横纵轴标签等
plt.xlabel('user_gender_id')
plt.ylabel('percent')
plt.title('user_gender&is_trade')
plt.show()
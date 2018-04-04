#coding:utf8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#装载数据
# data_csv = pd.read_table('round1_ijcai_18_train_20180301.txt',sep=' ')
data_csv = pd.read_table('round1_ijcai_18_test_a_20180301.txt',sep=' ')
data_origin = pd.DataFrame(data_csv)

##################################其他测试操作#####################################
#第一行的字段属性
print(data_origin[0:0])
# print("所有数据打印： ",data_origin)
# 478138条
print(len(data_origin))

# 按照某些字段去重，后面所有对源数据操作，以data_origin为准
# data_origin.drop_duplicates(['instance_id', 'item_id', 'item_category_list', 'item_property_list', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id', 'context_timestamp', 'context_page_id', 'predict_category_property', 'shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'is_trade'])
# print(len(data_origin))

# print(data_origin.describe())
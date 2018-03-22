#coding:utf8
import pandas as pd
import matplotlib as plt

# 当用行号索引的时候, 尽量用 iloc 来进行索引;
# 而用标签索引的时候用 loc ,

#装载数据
data_csv = pd.read_table('round1_ijcai_18_train_20180301.txt',sep=' ')
data_origin = pd.DataFrame(data_csv)

#第一行的字段属性
# print(data_origin[0:0])
# print("所有数据打印： ",data_origin) 478138条
# print(len(data_origin))

# 按照某些字段去重
data_origin.drop_duplicates(['instance_id', 'item_id', 'item_category_list', 'item_property_list', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id', 'context_timestamp', 'context_page_id', 'predict_category_property', 'shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'is_trade'])
# print(len(data_origin))

#统计字段is_trade的0和1的数据
a = len(data_origin[data_origin['is_trade']==0])
b = len(data_origin[data_origin['is_trade']==1])
#买/没买
# print(a/b)

print(data_origin['is_trade'].value_counts())


#########################################画图#########################################
#看看各乘客等级的获救情况
fig = plt.
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

#value_counts()计算Series里面某记录出现的频率
df=pd.DataFrame({u'买':a, u'未买':b})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")

plt.show()

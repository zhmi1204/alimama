#coding:utf8
import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re

#将unix时间戳value改为指定的format格式
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

#数据预处理
def convert_data(data):
    #将data里面的'context_timestamp'列属性换算成2018-09-09 13:59:59格式
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    #截取2018-09-09 13:59:59格式对应位置的数值
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    data['minute'] = data.time.apply(lambda x: int(x[14:16]))
    data['second'] = data.time.apply(lambda x: int(x[17:]))
    #不同user_id有197693条
    #groupby()方法能总结出不重复的按指定columns分组的记录，此处意为某用户在某一天的数据,共229465条
    #.size()方法能总结出groupby之后，数据出现的次数，也就是某用户在某一天浏览（或购买）过的商品次数
    #.reset_index()方法能给groupby.size之后的df重新设置索引，从零开始
    #.rename方法将size（）方法生成的列标签"0"改为user_query_day
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})

    #merge介绍url
    #https://blog.csdn.net/weixin_37226516/article/details/64137043
    #‘left’只的是以data里面的columns为基准对齐
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data


if __name__ == "__main__":
    #忽略警告
    warnings.filterwarnings("ignore")
    online = False# 这里用来标记是 线下验证 还是 在线提交

    data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)
    data.item_category_list = data.item_category_list.map(lambda x:re.split(';',str(x))[-1])
    #sort已弃用，要用sort_index或者sort_values
    data = data[['user_id','item_category_list','day','hour','minute','second']]
    #按照user_id排序，再在user_id用item_category_list内排序，然后用hour,minute,second等等
    data.sort_values(by=['user_id','item_category_list','day','hour','minute','second'], ascending=[0,1,2,3,4,5], inplace=True)

    #重置index
    data = data.reset_index(drop=True)
    print(data)
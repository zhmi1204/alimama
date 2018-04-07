#coding:utf8
import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings


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

    data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    data.hour[(data['hour'] >= 1)&(data['hour'] <= 4)] = int(1)
    data.hour[(data['hour'] >= 5)&(data['hour'] <= 8)] = int(2)
    data.hour[(data['hour'] >= 9)&(data['hour'] <= 12)] = int(3)
    data.hour[(data['hour'] >= 13)&(data['hour'] <= 16)] = int(4)
    data.hour[(data['hour'] >= 17)&(data['hour'] <= 20)] = int(5)
    data.hour[(data['hour'] >= 21)&(data['hour'] <= 23)] = int(6)
    data.hour[(data['hour'] == 0)] = int(6)

    data_1_is_trade1 = data[(data['hour'] == 1) & (data['is_trade'] == 1)]
    data_2_is_trade1 = data[(data['hour'] == 2) & (data['is_trade'] == 1)]
    data_3_is_trade1 = data[(data['hour'] == 3) & (data['is_trade'] == 1)]
    data_4_is_trade1 = data[(data['hour'] == 4) & (data['is_trade'] == 1)]
    data_5_is_trade1 = data[(data['hour'] == 5) & (data['is_trade'] == 1)]
    data_6_is_trade1 = data[(data['hour'] == 6) & (data['is_trade'] == 1)]

    data_1_is_trade0 = data[(data['hour'] == 1) & (data['is_trade'] == 0)]
    data_2_is_trade0 = data[(data['hour'] == 2) & (data['is_trade'] == 0)]
    data_3_is_trade0 = data[(data['hour'] == 3) & (data['is_trade'] == 0)]
    data_4_is_trade0 = data[(data['hour'] == 4) & (data['is_trade'] == 0)]
    data_5_is_trade0 = data[(data['hour'] == 5) & (data['is_trade'] == 0)]
    data_6_is_trade0 = data[(data['hour'] == 6) & (data['is_trade'] == 0)]

    plt.figure()
    #
    # 准备plt.bar数据，横轴数据，纵轴数据，label，放在底部的bottom
    data_divided = ['data_1', 'data_2', 'data_3', 'data_4', 'data_5', 'data_6']
    isbuy = [len(data_1_is_trade1), len(data_2_is_trade1), len(data_3_is_trade1),
             len(data_4_is_trade1), len(data_5_is_trade1), len(data_6_is_trade1),]
    nobuy = [len(data_1_is_trade0), len(data_2_is_trade0), len(data_3_is_trade0),
             len(data_4_is_trade0), len(data_5_is_trade0), len(data_6_is_trade0),]
    #
    # 画直方图
    plt.bar(data_divided, isbuy, label='isbuy')
    plt.bar(data_divided, nobuy, bottom=isbuy, label='nobuy')
    #
    # 坐标轴使用刻度
    plt.grid(True)
    # 覆盖plt.bar里面的横轴
    plt.xticks(data_divided)
    # 显示图例
    plt.legend(loc=0)
    # 显示横纵轴标签等
    plt.xlabel('data_divided')
    plt.ylabel('percent')
    plt.title('hours_divided_6&is_trade')
    plt.show()
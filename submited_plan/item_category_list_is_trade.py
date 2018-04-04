#coding:utf8
import time
import pandas as pd
import warnings
import matplotlib.pyplot as plt


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
    #忽略警告
    warnings.filterwarnings("ignore")
    online = False# 这里用来标记是 线下验证 还是 在线提交

    data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    #data['item_category_list'].value_counts()后获得如下结果：
    item_category_list = ['8277336076276184272','5755694407684602296','509660095530134768','5799347067982556520',
                          '7258015885215914736','2011981573061447208','8710739180200009128','3203673979138763595',
                          '2436715285093487584','8868887661186419229','4879721024980945592','1968056100269760729',
                          '6233669177166538628','22731265849056483']
    counts = [150775,102494,75418,72014,53041,9563,7084,2246,1966,1906,1207,186,123,88]

    isbuy = data.item_category_list[data.is_trade == 1].value_counts()/data.item_category_list.value_counts()
    nobuy = data.item_category_list[data.is_trade == 0].value_counts()/data.item_category_list.value_counts()

    data = pd.DataFrame({u'nobuy':nobuy,u'isbuy':isbuy})
    data.plot(kind = 'bar',stacked = True)
    plt.show()
#coding:utf8

#初始文件的操作，省略
# data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
# data.drop_duplicates(inplace=True)
# data = convert_data(data)

#将unix时间戳value改为指定的format格式
def timestamp_datetime(value):
    import time
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    return time.strftime(format, value)

#生成日期，mouth,day,hour等
def convert_data(data):
    # #将data里面的'context_timestamp'列属性换算成2018-09-09 13:59:59格式
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    # 截取2018-09-09 13：59：59格式对应位置的数值
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    # 不同user_id有197693条
    # groupby()方法能总结出不重复的按指定columns分组的记录，此处意为某用户在某一天的数据,共229465条
    # .size()方法能总结出groupby之后，数据出现的次数，也就是某用户在某一天浏览（或购买）过的商品次数
    # .reset_index()方法能给groupby.size之后的df重新设置索引，从零开始
    # .rename方法将size（）方法生成的列标签"0"改为user_query_day
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    # merge介绍url
    # https://blog.csdn.net/weixin_37226516/article/details/64137043
    # ‘left’只的是以data里面的columns为基准对齐
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])
    # 生成属性user_viewtimes
    user_viewtimes = data.groupby(['user_id', 'item_id']).size().reset_index().rename(columns={0: 'user_viewtimes'})
    data = pd.merge(data, user_viewtimes, 'left', on=['user_id', 'item_id'])
    data.user_viewtimes[data.user_viewtimes > 7] = int(8)
    # 生成属性user_id_item_category_list
    user_id_item_category_list = data.groupby(['user_id', 'item_category_list']).size().reset_index().rename(columns={0: 'user_id_item_category_list'})
    data = pd.merge(data, user_id_item_category_list, 'left', on=['user_id', 'item_category_list'])
    return data

#画图，按照每天is_buy/counts和no_buy/counts的比例
def perDayBuy(data):
    import matplotlib.pyplot as plt
    isbuy = data.day[data.is_trade == 1].value_counts() / data.day.value_counts()
    nobuy = data.day[data.is_trade == 0].value_counts() / data.day.value_counts()
    data = pd.DataFrame({u'nobuy': nobuy, u'isbuy': isbuy})
    data.plot(kind='bar', stacked=True)
    plt.xlabel('day')
    plt.ylabel('percent')
    plt.title('day_is_trade')
    plt.show()
    #补充：
    # data.day.value_counts()
    # 9月份
    # 18 78261
    # 21 71195
    # 19 70927
    # 20 68384
    # 22 68315
    # 23 63611
    # 24 57418
    # Name: day, dtype: int64

#训练数据，在线时生成baseline.csv，离线时输出log_loss到屏幕
def train_on_off_line(data,online = 'False'):
    import lightgbm as lgb
    from sklearn.metrics import log_loss
    import matplotlib.pyplot as plt
    # online = False  # 这里用来标记是 线下验证 还是 在线提交
    # True or False两种方式，一种生成.csv用于提交，由阿里妈妈官方评分，一种直接获取本地评分
    if online ==False:
        train=data.loc[data.day<24]   #18,19,20,21,22,23,24
        test=data.loc[data.day==24] #暂时先使用24天作为验证集
    elif online==True:
        train=data.copy()
        test=pd.read_csv('4.5_test_data.csv',sep=' ')
        test=convert_data(test)

    #选择训练的特征
    features = [ 'item_id','hour','item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level','user_gender_id','item_sees',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'shop_review_num_level', 'shop_id','shop_star_level','item_id_sees','train_predict_category_property_num',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description','item_category_list','user_viewtimes',
                ]
    #训练目标
    target=['is_trade']
    #fit与classifier的参数理解还是不太明白

    if online==False:
        clf=lgb.LGBMClassifier(num_leaves=32,max_depth=6,n_estimators=80,n_jobs=20,num_iterations=1000,learning_rate=0.01,feature_fraction= 0.58)
        clf.fit(train[features],train[target],feature_name=features,categorical_feature=['user_gender_id',
                                                                            'hour','item_sees', 'item_id_sees',])
        test['lgb_predict']=clf.predict_proba(test[features],)[:,1]
        print(log_loss(test[target],test['lgb_predict']))
        ax = lgb.plot_importance(clf, max_num_features=25)  #新加的属性重要性输出
        plt.show()
    else:
        clf = lgb.LGBMClassifier(num_leaves=32, max_depth=6, n_estimators=80, n_jobs=20, num_iterations=1000,
                                 learning_rate=0.01, feature_fraction=0.58)
        clf.fit(train[features],train[target],categorical_feature=['user_gender_id',
                                                                            'hour','item_sees', 'item_id_sees',])
        test['predicted_score']=clf.predict_proba(test[features])[:,1]
        test[['instance_id','predicted_score']].to_csv('4.6_1_baseline.csv',index=False,sep=' ') #保存在线的提交结果

#将24小时分成六个时段（1-4，5-8，9-12，13-16，17-20，21-0）
#每个时段买和不买的percent直方图
def hours_divided_6(data):
    import matplotlib.pyplot as plt
    data.hour[(data['hour'] >= 1) & (data['hour'] <= 4)] = int(1)
    data.hour[(data['hour'] >= 5) & (data['hour'] <= 8)] = int(2)
    data.hour[(data['hour'] >= 9) & (data['hour'] <= 12)] = int(3)
    data.hour[(data['hour'] >= 13) & (data['hour'] <= 16)] = int(4)
    data.hour[(data['hour'] >= 17) & (data['hour'] <= 20)] = int(5)
    data.hour[(data['hour'] >= 21) & (data['hour'] <= 23)] = int(6)
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
    # 准备plt.bar数据，横轴数据，纵轴数据，label，放在底部的bottom
    data_divided = ['data_1', 'data_2', 'data_3', 'data_4', 'data_5', 'data_6']
    isbuy = [len(data_1_is_trade1), len(data_2_is_trade1), len(data_3_is_trade1),
             len(data_4_is_trade1), len(data_5_is_trade1), len(data_6_is_trade1), ]
    nobuy = [len(data_1_is_trade0), len(data_2_is_trade0), len(data_3_is_trade0),
             len(data_4_is_trade0), len(data_5_is_trade0), len(data_6_is_trade0), ]
    # 画直方图
    plt.bar(data_divided, isbuy, label='isbuy')
    plt.bar(data_divided, nobuy, bottom=isbuy, label='nobuy')
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

#每小时的交易比值
def hour_is_trade(data):
    import matplotlib.pyplot as plt
    isbuy = []
    nobuy = []
    hours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    #手动统计出来的train数据里面每个hour的总浏览次数print(data['hour'].value_counts())
    counts = [10795,4486,2464,1889,1564,2336,5388,11014,15481,19925,24324,24082,24667,27497,27218,26868,24562,23450,25358,33733,41118,42751,35973,21168]
    for i in range(0,24):
        #/counts意为取比例
        isbuy.append(len(data[(data['hour'] == i)&(data['is_trade'] == 1)].index.tolist())/counts[i])
        nobuy.append(len(data[(data['hour'] == i)&(data['is_trade'] == 0)].index.tolist())/counts[i])
    # 画直方图
    plt.bar(hours, isbuy, label='isbuy')
    plt.bar(hours, nobuy, bottom=isbuy, label='nobuy')
    # 坐标轴使用刻度
    plt.grid(True)
    # 覆盖plt.bar里面的横轴
    plt.xticks(hours)
    # 显示图例
    plt.legend(loc=0)
    # 显示横纵轴标签等
    plt.xlabel('hours')
    plt.ylabel('trade_counts')
    plt.title('hours&is_trade')
    plt.show()

#item_category_list第二类（第一大类为7908382889764677758）里买和不买的比值
def item_category_list_is_trade(data):
    import matplotlib.pyplot as plt
    # data['item_category_list'].value_counts()后获得如下结果：
    item_category_list = ['8277336076276184272', '5755694407684602296', '509660095530134768', '5799347067982556520',
                          '7258015885215914736', '2011981573061447208', '8710739180200009128', '3203673979138763595',
                          '2436715285093487584', '8868887661186419229', '4879721024980945592', '1968056100269760729',
                          '6233669177166538628', '22731265849056483']
    counts = [150775, 102494, 75418, 72014, 53041, 9563, 7084, 2246, 1966, 1906, 1207, 186, 123, 88]

    isbuy = data.item_category_list[data.is_trade == 1].value_counts() / data.item_category_list.value_counts()
    nobuy = data.item_category_list[data.is_trade == 0].value_counts() / data.item_category_list.value_counts()

    data = pd.DataFrame({u'nobuy': nobuy, u'isbuy': isbuy})
    data.plot(kind='bar', stacked=True)
    plt.show()

#显示data里面item_property_list的value_counts()
def item_property_list_value_counts(data):
    import re
    prop_lst = data.item_property_list
    length = len(prop_lst)
    tmp = []
    for i in range(length):
        try:
            tmp += re.split(';', prop_lst[i])
            # print(i)
        except:
            continue
    return pd.Series(tmp).value_counts()

#按照标准的test将混乱的disordered进行排序
def sortDisodByInsIdOfTst(disorder_Path,test_path):
    data1 = pd.read_csv(disorder_Path, sep=' ')
    data2 = pd.read_csv(test_path, sep=' ')['instance_id']
    data_instance_id = []
    data_predicted_score = []
    for i in data2:
        data_instance_id.append(i)
        data_predicted_score.append(float(data1[data1['instance_id'] == i]['predicted_score']))
    c = {'instance_id': data_instance_id, 'predicted_score': data_predicted_score}
    #设置DataFrame的精确度
    pd.set_option('precision', 18)
    return pd.DataFrame(c)

#内排序模板
def standard_sort(data):
    import re
    data.item_category_list = data.item_category_list.map(lambda x:re.split(';',str(x))[-1])
    #sort已弃用，要用sort_index或者sort_values
    data = data[['user_id','item_category_list','day','hour','minute','second']]
    #按照user_id排序，再在user_id用item_category_list内排序，然后用hour,minute,second等等
    data.sort_values(by=['user_id','item_category_list','day','hour','minute','second'], ascending=[0,1,2,3,4,5], inplace=True)
    #重置index并弃用原有index
    return data.reset_index(drop=True)

#按照user/item_category_list/等顺序排序，生成frequency.txt文件
def sort1123412(data):
    import re
    data.item_category_list = data.item_category_list.map(lambda x: re.split(';', str(x))[-1])
    # sort已弃用，要用sort_index或者sort_values
    data = data[['user_id', 'item_category_list', 'day', 'hour', 'minute', 'second']]
    # 按照user_id排序，再在user_id用item_category_list内排序，然后用hour,minute,second等等
    data.sort_values(by=['user_id', 'item_category_list', 'day', 'hour', 'minute', 'second'],
                     ascending=[0, 1, 2, 3, 4, 5], inplace=True)
    f = []
    tmp_user_id = tmp_item_category_list = k = 0
    for row in data.iterrows():
        if row[1]['user_id'] != tmp_user_id:
            k = 1
            tmp_user_id = row[1]['user_id']
        else:  # 相同的user_id时
            if row[1]['item_category_list'] != tmp_item_category_list:
                k = 1
            else:  # 相同的item时
                k += 1
        tmp_item_category_list = row[1]['item_category_list']
        f.append(k)
        # count += 1
        # print(count)
    with open('frequency.txt', 'w')as file:
        file.write(str(f))
        # 经查验 f最大为9

#按照user/item_category_list/等顺序排序，生成destPath.csv文件
def sort122233(data,destPath):
    item_category_list_index_time = []
    data.sort_values(by=['user_id', 'item_category_list', 'context_timestamp'], ascending=[0, 1, 2], inplace=True)
    data = data.reset_index(drop=True)
    index_user = 1
    index_item = 1
    new_ss = []
    for row in data.iterrows():
        if (row[1]['user_id'] == index_user) and (row[1]['item_category_list'] == index_item):
            continue
        else:
            index_user = row[1]['user_id']
            index_item = row[1]['item_category_list']
            s = data[(data.user_id == row[1]['user_id']) & (data.item_category_list == row[1]['item_category_list'])]
            if len(s) == 1:
                new_ss.append(1)
            else:
                for new_s in range((len(s) - 1)):
                    new_ss.append(2)
                new_ss.append(3)
    data = pd.concat([data, pd.DataFrame({'item_sees': new_ss})], axis=1)
    data.to_csv(destPath, index=False, sep=' ')

#拼接train与test
def concat_train_test(trainPath,testPath):
    data1 = pd.read_csv(trainPath, sep=' ')
    data2 = pd.read_csv(testPath, sep=' ')
    data = pd.concat([data1, data2], axis=0)
    pd.set_option('precision', 18)
    data = data.reset_index(drop=True)
    return data

#显示data里面predict_category_property的value_counts()
def predict_category_property_value_counts(data):
    import re
    pred_lst = data.predict_category_property
    length = len(pred_lst)
    tmp = []
    for i in range(length):
        try:
            #同时按照:和;对字符串进行切割
            tmp += re.split(':|;|,', pred_lst[i])
        except:
            continue
    return pd.Series(tmp.remove('-1')).value_counts()

#本方法生成新特征position，用于标注item_category_list的第二个类目在predict_category_property里面第几个;前出现
def item_cate_lst_pos_predict_property(data):
    import re
    import matplotlib.pyplot as plt
    length = len(data.predict_category_property)
    position = []
    #注意逻辑
    count = 0
    for i in range(length):
        if count == 1:
            position.append(-1)
        count = 1
        f1 = re.split(';',data.predict_category_property[i])
        f2 = data.item_category_list[i]
        for j in f1:
            if str(f2) in j:
                position.append(f1.index(j))
                count = 0
                break
    return position
    #生成一个新的.csv源文件
    # newfile = pd.concat([data, pd.DataFrame(position, columns=['position'])], axis=1)
    # newfile.to_csv('4_new_test_1.csv',sep = ' ')


#生成item_cate_lst_pos_predict_property的position图像
def plot_position(data):
    import matplotlib.pyplot as plt
    isbuy = data.position[data.is_trade == 1].value_counts() / data.position.value_counts()
    nobuy = data.position[data.is_trade == 0].value_counts() / data.position.value_counts()

    data = pd.DataFrame({u'nobuy': nobuy, u'isbuy': isbuy})
    data.plot(kind='bar', stacked=True)
    plt.show()
############################################################################################################
#def
def test(data):
    a = data.loc(0)
    print(a)










###########################################################################################################
#注释
'''
1--def timestamp_datetime(value):       将unix时间戳value改为指定的format格式
2--def convert_data(data):              生成日期，mouth,day,hour等
3--def perDayBuy(data):                 画图，按照每天is_buy/counts和no_buy/counts的比例
4--def train_on_off_line(data):         训练数据，在线时生成baseline.csv，离线时输出log_loss到屏幕
5--def hours_divided_6(data):           将24小时分成六个时段（1-4，5-8，9-12，13-16，17-20，21-0）,每个时段买和不买的percent直方图
6--def hour_is_trade(data):             每小时的交易比值
7--def item_category_list_is_trade(data):
                                        item_category_list第二类（第一大类为7908382889764677758）里买和不买的比值
8--def item_property_list_value_counts(data):
                                        显示data里面item_property_list的value_counts()
9--def sortDisodByInsIdOfTst(disorder_Path, test_path):
                                        按照标准的test将混乱的disordered进行排序
10--def standard_sort(data):            内排序模板
11--def sort1123412(data):
                                        按照user/item_category_list/等顺序排序，生成frequency.txt文件
12--def sort122233(data, destPath):
                                        按照user/item_category_list/等顺序排序，生成destPath.csv文件
13--def concat_train_test(trainPath,testPath):
                                        拼接train与test
14--def predict_category_property_value_counts(data):
                                        显示data里面predict_category_property的value_counts()
15--def item_cate_lst_pos_predict_property(data):
                                        本方法生成新特征position，用于标注item_category_list的第二个
                                        类目在predict_category_property里面第几个;前出现
16--def plot_position(data):            生成item_cate_lst_pos_predict_property的position图像
'''

if __name__ == "__main__":
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")#忽略警告
    data = pd.read_csv('4_new_test_1.csv', sep=' ')

    ########################################################################################################
    #代码#
    # test(data)

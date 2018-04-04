#coding:utf8
import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
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
    #忽略警告
    warnings.filterwarnings("ignore")
    online = True #在线提交
    # online = False #线下验证

    data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)
    # 选择训练的特征
    features = ['item_price_level', 'item_sales_level','is_trade',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level','day',
                                                     'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                'context_page_id' ]
    df=data[features]

    # 训练目标
    target = ['is_trade']
    #

    # print(data)

    #True or False两种方式，一种生成.csv用于提交，由阿里妈妈官方评分，一种直接获取本地评分
    if online == False:
        train = df.loc[df.day < 24]  # 18,19,20,21,22,23,24
        test = df.loc[df.day == 24]  # 暂时先使用第24天作为验证集
        train.drop('day',axis=1,inplace=True)
        test.drop('day',axis=1,inplace=True)
    elif online == True:
        train = df.copy()
        test = pd.read_csv('round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = convert_data(test)

    feature= ['item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level',
                'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
              'context_page_id'  ]

#
    #fit与classifier的参数理解还不太明白
    #num_leaves=2^max_depth
    #参考链接：https://zhuanlan.zhihu.com/p/27916208
    if online == False:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[feature], train[target], feature_name=feature, categorical_feature=['user_gender_id', 'user_occupation_id',
                'user_age_level'])
        test['lgb_predict'] = clf.predict_proba(test[feature],)[:, 1]
        print(log_loss(test[target], test['lgb_predict']))
    else:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target],
                categorical_feature=['user_gender_id', ])
        test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
        test[['instance_id', 'predicted_score']].to_csv('baseline.csv', index=False,sep=' ')#保存在线提交结果
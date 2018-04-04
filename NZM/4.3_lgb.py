import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#将unix时间戳value改为指定的format格式
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value=time.localtime(value)
    dt=time.strftime(format,value)
    return  dt
#数据预处理
def convert_data(data):
    #对shop_id进行one_hot
    shop_id_labelencoder=LabelEncoder()
    shop_id_labelencoder.fit(data['shop_id'])
    data['shop_id']=shop_id_labelencoder.transform(data['shop_id'])
    #对city进行one_hot
    shop_id_labelencoder = LabelEncoder()
    shop_id_labelencoder.fit(data['item_city_id'])
    data['item_city_id'] = shop_id_labelencoder.transform(data['item_city_id'])
    #对brand进行one_hot
    shop_id_labelencoder = LabelEncoder()
    shop_id_labelencoder.fit(data['item_brand_id'])
    data['item_brand_id'] = shop_id_labelencoder.transform(data['item_brand_id'])

    data['item_category_list'] = data['item_category_list'].map(lambda x: int(str(x).split(';')[1]))
    # #将data里面的'context_timestamp'列属性换算成2018-09-09 13:59:59格式
    data['time']=data.context_timestamp.apply(timestamp_datetime)
    #截取2018-09-09 13：59：59格式对应位置的数值
    data['day']=data.time.apply(lambda x:int(x[8:10]))
    data['hour']=data.time.apply(lambda x:int(x[11:13]))
    # 不同user_id有197693条
    # groupby()方法能总结出不重复的按指定columns分组的记录，此处意为某用户在某一天的数据,共229465条
    # .size()方法能总结出groupby之后，数据出现的次数，也就是某用户在某一天浏览（或购买）过的商品次数
    # .reset_index()方法能给groupby.size之后的df重新设置索引，从零开始
    # .rename方法将size（）方法生成的列标签"0"改为user_query_day
    user_query_day=data.groupby(['user_id','day']).size().reset_index().rename(columns={0:'user_query_day'})
    # merge介绍url
    # https://blog.csdn.net/weixin_37226516/article/details/64137043
    # ‘left’只的是以data里面的columns为基准对齐
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_our=data.groupby(['user_id','day','hour']).size().reset_index().rename(columns={0:'user_query_day_hour'})
    data=pd.merge(data,user_query_day_our,'left',on=['user_id','day','hour'])
    #生成属性user_viewtimes
    user_viewtimes =data.groupby(['user_id', 'item_id']).size(
    ).reset_index().rename(columns={0: 'user_viewtimes'})
    data = pd.merge(data, user_viewtimes, 'left', on=['user_id', 'item_id'])
    data.user_viewtimes[data.user_viewtimes > 7] = int(8)
    #生成属性user_id_item_category_list
    user_id_item_category_list =data.groupby(['user_id', 'item_category_list']).size(
    ).reset_index().rename(columns={0: 'user_id_item_category_list'})
    data = pd.merge(data, user_id_item_category_list, 'left', on=['user_id', 'item_category_list'])



    return data


if __name__=="__main__":
    #忽略警告
    warnings.filterwarnings("ignore")
    online=False#这里用来标记是线下验证还是在线验证
    data=pd.read_csv("new_train.csv",sep=' ')
    data.drop_duplicates(inplace=True)

    data=convert_data(data)

    #print(data)
    #true和falese 两种方式，一种生成用于.csv用于提交，由阿里妈妈官方进行评分，一种直接获取本地的评分
    if online ==False:
        train=data.loc[data.day<24]   #18,19,20,21,22,23,24
        test=data.loc[data.day==24] #暂时先使用24天作为验证集
    elif online==True:
        train=data.copy()
        test=pd.read_csv('new_test.csv',sep=' ')
        test=convert_data(test)

    #选择训练的特征
    features = [ 'item_id','hour','item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level','user_gender_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'shop_review_num_level', 'shop_id','shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                'train_predict_category_property_num','item_category_list','user_viewtimes',
                ]
    #训练目标
    target=['is_trade']
    #fit与classifier的参数理解还是不太明白

    if online==False:
        clf=lgb.LGBMClassifier(num_leaves=32,max_depth=6,n_estimators=80,n_jobs=20,num_iterations=1000,learning_rate=0.01,feature_fraction= 0.55)
        clf.fit(train[features],train[target],feature_name=features,categorical_feature=['user_gender_id',
                                                                            'train_predict_category_property_num','hour',
                                                                                         ])
        test['lgb_predict']=clf.predict_proba(test[features],)[:,1]
        print(log_loss(test[target],test['lgb_predict']))
        ax = lgb.plot_importance(clf, max_num_features=25)  #新加的属性重要性输出
        plt.show()
    else:
        clf = lgb.LGBMClassifier(num_leaves=32, max_depth=6, n_estimators=80, n_jobs=20, num_iterations=1000,
                                 learning_rate=0.01, feature_fraction=0.6)
        clf.fit(train[features],train[target],categorical_feature=['user_gender_id','train_predict_category_property_num','hour',])
        test['predicted_score']=clf.predict_proba(test[features])[:,1]
        test[['instance_id','predicted_score']].to_csv('4.3_3new_baseline.csv',index=False,sep=' ') #保存在线的提交结果
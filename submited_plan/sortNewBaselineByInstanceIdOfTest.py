#coding:utf8
import pandas as pd
import warnings

if __name__ == "__main__":
    #忽略警告
    warnings.filterwarnings("ignore")
    online = False# 这里用来标记是 线下验证 还是 在线提交

    data1 = pd.read_csv('4.4_1_new_baseline.csv', sep=' ')
    data2 = pd.read_csv('round1_ijcai_18_test_a_20180301.txt', sep=' ')['instance_id']
    data_instance_id = []
    data_predicted_score = []

    for i in data2:
        data_instance_id.append(i)
        data_predicted_score.append(float(data1[data1['instance_id'] == i]['predicted_score']))
    c = {'instance_id': data_instance_id, 'predicted_score': data_predicted_score}
    pd.set_option('precision', 18)
    data = pd.DataFrame(c)
    print(data)
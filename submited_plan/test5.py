import pandas as pd
import re

data = pd.read_csv('3_new_test_1.csv',sep = ' ')
tmp = data[data.instance_id == 4033950247523642138 ].predict_category_property
# ls = re.split(';',str(tmp))
# print(ls[0:10])
print(list(tmp))
# 8277336076276184272
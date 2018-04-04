#coding:utf8
import pandas as pd
import numpy as np
import re

def read_freq():
    f = open('frequency.txt','r')
    file = f.read()
    condition = lambda t: t != ","
    file = list(filter(condition,file))
    file.pop(-1)
    file.pop(0)
    f.close()

    list = []
    for i in range(len(file)):
        if file[i] == ' ':
            continue
        else:
            list.append(int(file[i]))
    return list
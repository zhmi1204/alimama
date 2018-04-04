# coding:utf8
import pandas as pd
import numpy as np
import re

def read_freq():
    f = open('frequency.txt', 'r')
    file = f.read()
    condition = lambda t: t != ","
    file = list(filter(condition, file))
    file.pop(-1)
    file.pop(0)
    f.close()

    freq = []
    for i in range(len(file)):
        if file[i] == ' ':
            continue
        else:
            freq.append(int(file[i]))
    return freq

if __name__ =="__main__":
    freq = read_freq()
    tmp = 0
    for i in freq:
        if i > tmp:
            tmp = i
    print(tmp)
    pass
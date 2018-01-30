# -*- coding:utf-8 -*-
__author__ = 'leon'
from pandas import Series, DataFrame
import pandas as pd


a = Series([100, 'aaa', 'SSS'])
#print(a, a.values, a.index)
# 自定义index
a = Series([100, 'aaa', 'bbb'], index=['name', 'address', 3333])
a['name'] = 200
#print(a['name'])
# 可以将一个dict转换为series
a = {'python': 80, 'c++': 70, 'java': '60'}
#print(Series(a))
#list转换为series
my_list = ['java', 'c++']
a = Series(a, index=my_list)
#print(a)
# dict转换DataFrame
data = {'name': ['yahoo','google','facebook'], 'marks':[200, 400, 600], 'price':[1,2,3]}
f1 = DataFrame(data)
#print(f1)
# 可以更换列columns名称
f2 = DataFrame(data, columns=['name', 'price', 'marks'])
#print(f2)
f3 = DataFrame(data, columns=['name','price','marks','debt'], index=['a','b','c'])
print(f3)
print(f3.columns)
print(f3['name'])

# 可以使用dict套dict的方式
newdata = {'lang': {'first': 'python', 'second': 'c++'}, 'price': {'first':'java'}}
#print(DataFrame(newdata))





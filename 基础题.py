#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""               第一部分 字符串
https://blog.csdn.net/qq_14945431/article/details/78752063    """

"""
遇到字符串的题，多用用切片，+，和一些现成的函数。骚操作太多啦
"""

#1. 已知字符串 a = "aAsmr3idd4bgs7Dlsf9eAF",要求如下
#
a = "aAsmr3idd4bgs7Dlsf9eAF"

#1.1 请将a字符串的大写改为小写，小写改为大写。
#
a = "aAsmr3idd4bgs7Dlsf9eAF"
print a.swapcase()

#1.2 请将a字符串的数字取出，并输出成一个新的字符串。str.join(sequence)，sequence -- 要连接的元素序列。
#
a = "aAsmr3idd4bgs7Dlsf9eAF"

print ''.join([s for s in a if s.isdigit()]) # 用了个列表生成式

#1.3 请统计a字符串出现的每个字母的出现次数（忽略大小写，a与A是同一个字母），并输出成一个字典。 例 {'a':4,'b':2}
#
a = "aAsmr3idd4bgs7Dlsf9eAF"

a = a.lower() # 先将大写转为小写
print dict([( x,a.count(x) ) for x in set(a)]) # 先获取list，再转换为dict

#1.4 请去除a字符串多次出现的字母，仅留最先出现的一个。例 'abcabb'，经过去除后，输出 'abc'
#
a = "aAsmr3idd4bgs7Dlsf9eAF"
a_list = list(a) #转换成list
set_list = list(set(a_list)) #去重以后再转换回list，但字母出现的顺序可能发生了改变。
set_list.sort(key=a_list.index)
# key参数来指定一个函数，此函数将在每个元素比较前被调用
# 对去重以后的list按照原先的索引（顺序）进行排序，reverse = False 升序（默认）
print ''.join(set_list)#拼接成字符串

#1.5 请将a字符串反转并输出。例：'abc'的反转是'cba'
#
a = "aAsmr3idd4bgs7Dlsf9eAF"
print a[::-1] #用切片来做，[0:3]就代表取出索引为0,1,2的三个元素，-1代表从最后一个元素开始（间隔为1）

#1.6 去除a字符串内的数字后，请将该字符串里的单词重新排序（a-z），并且重新输出一个排序后的字符串。
# （保留大小写,a与A的顺序关系为：A在a前面。例：AaBb）
'''
1.要有小写字母从a-z的排序
2.大小写不同，但值相同的字母,大写在小写的前面
思路：先整体排序，然后遍历字符串，将大写字母和小写字母分别放到一个list中
将大写字母转为小写字母，若存在对应的小写字母，则将这个大写字母放到小写字母序列对应的字母位置前面
'''
a = "aAsmr3idd4bgs7Dlsf9eAF"
l = sorted(a)
# sort本地排序，sorted返回副本，原始输入不变
# 需要注意：sort()方法仅定义在list中，而sorted()方法对所有的可迭代序列都有效
#  并且针对任何的可迭代序列，sorted()都是返回一个list，
# 默认排序的key：ASCII码排序:数字，大写字母，小写字母
# 将大写字母，小写字母以及其他字符放到对应list中
a_upper_list = []
a_lower_list = []
for x in l:
    if x.isupper():
        a_upper_list.append(x)
    elif x.islower():
        a_lower_list.append(x)
    else:
        pass
# 将大写字母转为小写字母，若存在对应的小写字母，则将这个大写字母放到小写字母前面的位置
for y in a_upper_list:
    y_lower = y.lower()
    if y_lower in a_lower_list:
        a_lower_list.insert(a_lower_list.index(y_lower),y)
""" insert(位置，某字符)：a_lower_list.index(y_lower) 即 y_lower在a_lower_list中对应的索引"""
print a_lower_list  # 这是由一个个字符组合成的list
print ''.join(a_lower_list) # 转为字符串输出

#1.7 请判断 'boy'里出现的每一个字母，是否都出现在a字符串里。如果出现，则输出True，否则，则输 出False.
#
a = "aAsmr3idd4bgs7Dlsf9eAF"
search = 'boy'
u = set(a)
u.update(list(search))
# set的update() 方法用于修改当前集合，可以添加新的元素到当前集合中，如果添加的元素在集合中已存在，则忽略该元素
# 若set(a)中的元素个数==u中元素个数，说明a中包含了boy三个字母
print len(set(a)) == len(u)

# 另一种方法：
# x.issubset(y)：判断集合x的所有元素是否都包含在集合y中，如果都包含则返回True
a = "aAsmr3idd4bgs7Dlsf9eAF"
print set('boy').issubset(set(a))

#1.8 要求如1.7，此时的单词判断，由'boy'改为四个，分别是 'boy','girl','bird','dirty'，判断这4个字符串里的每个字母是否都出现在a中
#
a = "aAsmr3idd4bgs7Dlsf9eAF"
lst = ['boy','girl','bird','dirty']
s = ''.join(lst) # 将这四个字符串合并为一个字符串
print set(s).issubset(set(a))

#1.9 输出a字符串出现频率最高的字母
a = "aAsmr3idd4bgs7Dlsf9eAF"

# 构造一个dict，key：字母，value：字母出现个数
l = ([(x,a.count(x)) for x in set(a)]) # 里面用了个[]，代表列表生成式
# collections.Counter 函数可以将数组中每个数值以及它们对应的数量生成一个dict
l.sort(key = lambda k:k[1],reverse=True)
# list.sort()和sorted()函数使用key参数来指定一个函数，此函数将在每个元素比较前被调用
# lambda k:k[1]是一个(匿名)函数，即 此时取出元素的[1]即它的频次进行排序
# set_list.sort(key=a_list.index)
print l[0][0],l[0][1] # 输出出现频次最高的元素以及它出现的次数，dict[0][0],dict[0][1]


#2.在python命令行里，输入import this 以后出现的文档，统计该文档中，"be" "is" "than" 的出现次数。
#

import os
m = os.popen('python -m this').read()
m = m.replace('\n','')
l = m.split(' ')
print [(x,l.count(x)) for x in ['be','is','than']]


#3.一文件的字节数为 102324123499123，请计算该文件按照kb与mb计算得到的大小。
size = 102324123499123
print '%s kb'%(size >> 10) # 2^10字节==kb；2^20字节==mb
print '%s mb'% (size >> 20)


#4.已知 a = [1,2,3,6,8,9,10,14,17],请将该list转换为字符串，例如 '123689101417'.
#
a = [1,2,3,6,8,9,10,14,17]
list1=list()
list1=map(str,a) # 将a中每一个元素转为字符
print ''.join(list1)


#5.其他

b=[23,45,22,44,25,66,78]
print [i for i in b if i % 2==1]
print [m+2 for m in b]
print range(11,34,11) #range(起始，结束+1，步长)前两个都是索引
""" 由于tuple是不可变的，要改变其中的元素，可以首先将其变为list改变元素，再用tuple()函数转为元祖 """

"""
翻转单词顺序：输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。student. a am I
"""
# 思路：第一步翻转句子中所有的字符，第二步再翻转每个单词中字符的顺序，通过扫描空格来确定每个单词的起始和终止位置
# python:首先通过空格切割字符串，每一个单词作为list中一项，join函数将list中的每个单词用空格连接起来
# 并且仅仅对list中每个元素（即每个单词）倒序排列，即倒着摆放
"""
['what', 'the', 'fuck.']
fuck. the what
"""


# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        l = s.split(' ')  # 以空格来分割单词
        return ' '.join(l[::-1])  # 起点坐标，终点坐标向正或负方向多走一步，方向


"""
左旋转字符串：对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
思路：和上面的例题有关，可以把原字符串当做两部分，直接切开然后拼接起来就好
"""


# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        return s[n:] + s[:n]



"""               第二部分 基础编程题
https://blog.csdn.net/qq_31796711/article/details/78924297      """


#2、输出“水仙花数”。所谓水仙花数是指1个3位的十进制数，其各位数字的立方和等于该数本身。
#   例如：153是水仙花数，因为153 = 13 + 53 + 33
for i in range(100,1000):
    a=str(i)
    b=int(a[0])
    c=int(a[1])
    d=int(a[2])
    if i==pow(b,3)+pow(c,3)+pow(d,3):
        print(i)

# 4、求200以内能被17整除的最大正整数。
for i in range(200,17,-1):
    if i%17==0:
        print i,i/17
        break

# 10、在屏幕上打印1-30数，每7个数换行
for i in range(1,31):
     if i%7 == 1:
        print '\n'
     print i,  # 这样就不会换行了
print '\n'

# 11、打印1000以内的所有素数(质数,只能被1和自身整除的数)
i = 2
while(i < 1000):
    j = 2
    while(j < i): # 要缩小循环范围 可以改为 j<=(i/j)
        if not(i%j):  # 若能整除j则开始下一个i值循环
            break
        j = j + 1
        if (j == i) :  # 这里对应改为 j>(i/j)
            print i,
    i = i + 1
print '\n'

# 13、九九乘法口诀表
"""  逐行输出：
1*1=1	

1*2=2	2*2=4	可以看出，可以写成 j*i 的形式，其中j取得1到i的所有数，j<=i，取j=range(1,i+1),i取range(1,10)

1*3=3	2*3=6	3*3=9	   每次更新i时（外循环），先跳行

"""
for i in range(1,10):
    for j in range(1,i+1):
        print "%d*%d=%d\t"%(j,i,j*i),
    print '\n'
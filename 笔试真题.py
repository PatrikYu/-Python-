#  coding: utf-8
from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""
基本输入操作
"""
import sys
a = input("input a: ")  # 只能输入一个数字
b1 = raw_input("raw_input b1: ") # 字符串
b2 = raw_input("raw_input b2: ").split() # 以空格切割字符串，得到一个list
c1 = sys.stdin.readline()
c2 = sys.stdin.readline().strip('\n').split() # 以空格切割字符串，得到一个list
print(a,type(a))
print(b1,type(b1))
print(b2,type(b2))
print(c1,type(c1))
print(c2,type(c2))

m1 = sys.stdin.readlines() # 以<CTRL-D>结束输出,结果为list，注意ctrl+D必须在下一行的回车出去打
m2 = m1[0].strip().split()
m3 = m1[1].strip().split()
print m1
print m2,m3,
# 逗号以空格相隔,而且m3后面有逗号，接下来的m3也是空一个空格输出的，不会换行
print m3


"""
例1：读入多行数据（没告知行数），输出每一行数据的前两项之和
3 2 
5 4 1
"""
import sys
while True:
    line = sys.stdin.readline() # 一次只读一行   # line = raw_input()
    if not line.strip('\n'): # 如果是空行(^Z)就停止  # if not line:
        break
    a = line.split()
    print int(a[0]) + int(a[1]) # 否则回显，再回去读下一行

"""
例2：输入矩阵的阶数n以及这个矩阵，输出一个整数，表示n阶方阵的和。
3
1 2 3
2 1 3
3 2 1
"""

import sys
# 读取输入的第一行，先去掉换行符，再转为整型
n = int(sys.stdin.readline().strip()) # strip函数默认丢掉 \n
ans = 0
for i in range(n):
    # n就是即将读入的矩阵的行数，这样就不用判断什么时候结束读取了
    # 读取每一行，并将其变为列表，split函数默认以空格分隔
    line = sys.stdin.readline().strip().split()
    # 将此列表中每个元素从str类型变为int类型，依然为list
    values = map(int, line)
    for v in values:
        ans += v
print ans

"""
腾讯真题，拿球
"""
#-*- coding:utf-8 -*-
def onetake(n,a,b):
    sum_a = 0
    sum_b = 0
    if n == 1:
        return a[0]
    while n:
        index_a = a.index(max(a))   # max函数与index函数都只保存遇到的第一个
        sum_a += a.pop(index_a) # 弹出最大值,并累加到sum中
        index_b = b.index(max(b))
        b.pop(index_b)  # 弹出最大值
        n -= 1
        if not n:
            return sum_a - sum_b
        index_b = b.index(max(b))
        sum_b += b.pop(index_b)
        index_a = a.index(max(a))
        a.pop(index_a)
        n -= 1
    return sum_a-sum_b
#
# print onetake(3,[5,3,8],[6,7,7])

"""
快手：第一题：对于一个长度为n的整数序列，其中只有一个数字的出现次数为奇数，找出来，输入第一行为数字个数，第二行为读入序列
9
2 2 2 3 5 8 3 8 5
"""
import sys
import collections
n = int(sys.stdin.readline().strip())
list1 = sys.stdin.readline().strip().split() # 得到一个list，元素为字符型
c = collections.Counter(list1) # 统计每个字符出现的次数
for k, v in c.items():
    if v % 2==1:
        print k

"""
快手第二题：对于一个长度为n的整数序列，检查这个序列是否为非递减序列，输入第一行为数字个数，第二行为整数序列
非递减序列的定义是：array[i]<=array[i+1]
10
1 2 3 3 4 5 6 6 7 8
"""
import sys
n = int(sys.stdin.readline().strip())
temp = sys.stdin.readline().strip().split() # 得到一个list，元素为字符型
list1 = map(int,temp) # 变为整数list
for i in range(n):
    if i == n - 1:
        print True
        break
    if list1[i] > list1[i+1]:
        print False
        break

"""
蘑菇街1：输入第一行一个字符串，如：“1234567890”
输入第二行一个数字是n，如5
输出所有长度为n的子串，如“12345”，“23456”，“34567”，“45678”，“56789”
"""

strs = raw_input()
n = input()
if len(strs) < n or n < 0:  # 注意考虑n<0的情况，否则有0.2的demo无法通过
    print -1
else:
    for i in range(len(strs) - n + 1):
        print strs[i:i + n],  # 别人的方法是真的简单，直接输出str上某些索引的值啦


#-*- coding:utf-8 -*-
import sys
a = sys.stdin.readline().strip() # 字符串
n = int(sys.stdin.readline().strip())
if n>len(a) or n<0:   # 注意考虑n<0的情况
    print -1
else:
    temp=list()
    for j in range(len(a)-n+1):
        start = a[j]
        end = a[j+n-1]
        for i in range(j,j+n):
            temp.append(a[i])
        print ''.join(temp),
        temp = []

"""
蘑菇街2：两个单向有序链表的合并
第一行一个链表，如1 2 3 4 5
第二行一个链表，如2 3 4 5 6
输出：1 2 2 3 3 4 4 5 5 6
"""
#-*- coding:utf-8 -*-
import sys
temp = sys.stdin.readline().strip()
l1 = map(int,temp)
temp = sys.stdin.readline().strip()
l2 = map(int,temp)
i = 0
j = 0
l = list()
m = len(l1)
n = len(l2)
while m and n:
    if l1[i] <=l2[j]:
        l.append(l1[i])
        i += 1
        m -= 1
    else:
        l.append(l2[j])
        j += 1
        n -= 1
if m:
    l.append(l2[j:])
elif n:
    l.append(l1[i:])
print l

"""
网易18年1：
找出符合条件的正整数数对(x, y)个数，条件：x和y均不大于n, 并且x除以y的余数大于等于k。
输入包括两个正整数n,k(1 <= n <= 10^5, 0 <= k <= n - 1)。
5 2
"""
import sys
temp = sys.stdin.readline().strip().split()
n = int(temp[0])
k = int(temp[1])
num = 0
for i in range(1,n+1):
    for j in range(1,n+1):
        if i%y>=k:
            num += 1
print num
# 最讨厌这种找规律的题目了。。。。。。找不出来
n,k = map(int, raw_input().split())
result = 0
if(k == 0):
    print(n*n)
else:
    for chushu in range(k+1, n+1):
        result = result + (chushu - k)*(n // chushu)
        if (n % chushu) >= k:
            result = result + (n % chushu) - k + 1
    print(result)

"""
头条2018 1：
P为给定的二维平面整数点集。定义 P 中某点x，如果x满足 P 中任意点都不在 x 的右上方区域内（横纵坐标都大于x），则称其为“最大的”。
求出所有“最大的”点的集合。（所有点的横坐标和纵坐标都不重复, 坐标轴范围在[0, 1e9) 内）
如下图：实心点为满足条件的点的集合。请实现代码找到集合 P 中的所有 ”最大“ 点的集合并输出。
哎呀应该仔细观察它图中给出的红色的点：纵坐标最大的一定是res，然后找出横坐标比它大的就ok了（注意是在纵坐标依次减小的序列中去找）
"""
#-*- coding:utf-8 -*-
import sys
n = input()
point = [] # 输入的点
for i in range(n):
    point.append(map(int, sys.stdin.readline().strip().split())) # 如果不行，在map前面加个list函数
point.sort(key=lambda k: k[1], reverse=True) # 按点的纵坐标从高到低排序

res = [] # 输出的点
res.append(point[0]) # 纵坐标最大的点一定属于res
for i in range(1, len(point)): # 接下来比较它们的横坐标，后续加入到res中的必定满足横坐标大于res中的倒数第一个点（即他们中横坐标最大的点）
    if point[i][0] > res[-1][0]:
        res.append(point[i])
    else:
        continue
res.sort(key=lambda k: k[0]) #
for i in res:
    print i[0], i[1]

"""
leetcode84 Largest Rectangle in Histogram 
"""

class Solution:
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """

        heights.append(0)
        stack = []
        i = 0
        result = 0
        while i < len(heights):
            if not stack or heights[stack[-1]] < heights[i]:
                stack.append(i)
                i += 1
            else:
                num = stack.pop(-1)
                result = max(result, heights[num] * (i - stack[-1] - 1 if stack else i))
        return result

"""
给定一个数组序列, 需要求选出一个区间, 使得该区间是所有区间中经过如下计算的值最大的一个：
区间中的最小数 * 区间所有数的和最后程序输出经过计算后的最大值即可，不需要输出具体的区间。
如给定序列  [6 2 1]则根据上述公式, 可得到所有可以选定各个区间的计算值:
[6] = 6 * 6 = 36;
[2] = 2 * 2 = 4;
[1] = 1 * 1 = 1;
[6,2] = 2 * 8 = 16;
[2,1] = 1 * 3 = 3;
[6, 2, 1] = 1 * 9 = 9;
从上述计算可见选定区间 [6] ，计算值为 36， 则程序输出为 36。
区间内的所有数字都在[0, 100]的范围内;
"""
#-*- coding:utf-8 -*-
n=int(input())
arr=[int(x) for x in input().split()]
stack = []
arr.append(0) # 最后添加一个0用于出栈
result = 0
i = 0
presum = []
tempsum = 0
while i<len(arr):
    if not stack or arr[i]>=stack[-1]:
        presum.append(tempsum)
        tempsum = 0
        stack.append(arr[i])
        i+=1
    else:
        temp = stack.pop(-1)
        tempsum+=(temp+presum.pop())
        result = max(tempsum*temp,result)
print(result)

"""
最大矩阵问题：给定一个只含有0和1的矩阵图，找到其中仅含有1的矩阵，并且返回它的面积
"""


class Solution:
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if len(matrix) == 0:
            return 0

        row = len(matrix)
        column = len(matrix[0])
        temp = [0 for i in range(column)]
        i = 0
        result = 0
        while i < row:
            for j in range(column):
                temp[j] = (0 if matrix[i][j] == '0' else temp[j] + 1)
            result = max(result, self.largestRectangleArea(temp))
            i += 1
        return result

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.append(0)
        stack = []
        i = 0
        result = 0
        while i < len(heights):
            if not stack or heights[stack[-1]] < heights[i]:
                stack.append(i)
                i += 1
            else:
                num = stack.pop(-1)
                result = max(result, heights[num] * (i - stack[-1] - 1 if stack else i))
        return result

"""
百度19.1：
abaca 把最后一个字符放到第一行去，不断重复，判断是否有重复的
"""
# n = input()
# a = list(sys.stdin.readline().strip())
n =4
a = 'abab'
list1 = []
list1.append(a)
for i in range(n-1):
    # temp = a.pop()
    # a.insert(0, temp)
    # list1.append(a)  # 由于python采用地址引用，这样每次都会改变a，最终得到的list中元素完全相同
    b = a[n-1] + a[:n-1] # 字符串专用的+：表示连接；若是list的连接： b = a[n-1:n-1]+a[:n-1] 前面得到的依然是list，否则是字符
    list1.append(b)
    a = b
print len(set(list1))

"""
百度19.2：KMP算法用于返回子串匹配上的第一个位置
          字符串的count方法用于统计 某字符串 在一长串字符串 中的个数
          s= 'abababab'
          print s.count('aba')   
"""
# -*- coding:utf-8 -*-
import sys

def kmp(mom_string, son_string):
    # 传入一个母串和一个子串
    # 返回子串匹配上的第一个位置，若没有匹配上返回-1
    test = ''
    if type(mom_string) != type(test) or type(son_string) != type(test):
        return -1
    if len(son_string) == 0:
        return 0
    if len(mom_string) == 0:
        return -1
    # 求next数组
    next = [-1] * len(son_string)
    if len(son_string) > 1:  # 这里加if是怕列表越界
        next[1] = 0
        i, j = 1, 0
        while i < len(son_string) - 1:  # 这里一定要-1，不然会像例子中出现next[8]会越界的
            if j == -1 or son_string[i] == son_string[j]:
                i += 1
                j += 1
                next[i] = j
            else:
                j = next[j]

    # kmp框架
    m = s = 0  # 母指针和子指针初始化为0
    while (s < len(son_string) and m < len(mom_string)):
        # 匹配成功,或者遍历完母串匹配失败退出
        if s == -1 or mom_string[m] == son_string[s]:
            m += 1
            s += 1
        else:
            s = next[s]

    if s == len(son_string):  # 匹配成功
        return m - s
    # 匹配失败
    return -1

A = raw_input()
B = raw_input()
m = input()
a = []
for i in range(m):
    temp = map(int,sys.stdin.readline().strip().split())
    mom_string = A[temp[0],temp[1]]
    son_string = B
    print temp
    print kmp(mom_string, son_string)
"""
发现了没有，其实题目不难，第二天都能做出来，只不过自己练习得太少，一些常用的用法还不熟悉
其实如果昨晚百度的话应该还是能写出来的。多写题，这些用法才会熟悉起来
"""

"""
拼多多18.1：
给定一个无序数组，包含正数、负数和0，要求从中找出3个数的乘积，使得乘积最大，要求时间复杂度：O(n)，空间复杂度：O(1)
"""
# -*- coding:utf-8 -*-
n = input
a = map(int,raw_input().split())
a.sort()
if n > 2:
    max1 = a[-1]
    max2 = a[-2]
    max3 = a[-3]
    min1 = a[0]
    min2 = a[1]
    print max( max1*max2*max3 , min1*min2*max1 )  # 这种规律谁找得出来啊。。。

"""
有两个用字符串表示的非常大的大整数,算出他们的乘积，也是用字符串表示。不能用系统自带的大整数类型。
输入输出均以字符串表示
# python就不管它要求了，直接乘起来就好
"""
a = map(int,raw_input().split())
print str(a[0]*a[1])

"""
六一儿童节，老师带了很多好吃的巧克力到幼儿园。每块巧克力j的重量为w[j]，
对于每个小朋友i，当他分到的巧克力大小达到h[i] (即w[j]>=h[i])，他才会上去表演节目。
老师的目标是将巧克力分发给孩子们，使得最多的小孩上台表演。
可以保证每个w[i]> 0且不能将多块巧克力分给一个孩子或将一块分给多个孩子。

"""
"""
贪心算法 + while循环！！！
"""
# -*- coding:utf-8 -*-
# 就用贪心算法来做
n = input() # 人数
h = map(int,raw_input().split())
m = input() # 物品数
w = map(int,raw_input().split())
h.sort()
w.sort()
people = 0
i = 0
j = 0
while i<n and j<m:
    if h[i]<=w[j]:
        people += 1
        i += 1
        j += 1
    else:
        j += 1
print people

"""
在商城的某个位置有一个商品列表，该列表是由L1、L2两个子列表拼接而成。当用户浏览并翻页时，需要从列表L1、L2中获取商品进行展示。

1. 用户可以进行多次翻页，用offset表示用户在之前页面已经浏览的商品数量，比如offset为4，表示用户已经看了4个商品

2. n表示当前页面需要展示的商品数量

3. 展示商品时首先使用列表L1，如果列表L1长度不够，再从列表L2中选取商品

4. 从列表L2中补全商品时，也可能存在数量不足的情况

请根据上述规则，计算列表L1和L2中哪些商品在当前页面被展示了

每个测试输入包含1个测试用例，包含四个整数，分别表示偏移量offset、待摆放的元素数量n，列表L1的长度l1，列表L2的长度l2。

在一行内输出四个整数分别表示L1和L2的区间start1，end1，start2，end2，每个数字之间有一个空格。
注意，区间段使用半开半闭区间表示，即包含起点，不包含终点。如果某个列表的区间为空，使用[0, 0)表示，
如果某个列表被跳过，使用[len, len)表示，len表示列表的长度。

输入
2 4 4 4
1 2 4 4
4 1 3 3
输出
2 4 0 2
1 3 0 0
3 3 1 2
"""

# -*- coding:utf-8 -*-
a = map(int,raw_input().split())
offset = a[0]
n = a[1]
l1 = a[2]
l2 = a[3]
# 在草稿纸上把所有可能的情况都列一列，仔细思考画图，不要遗漏了区间
# 比较 offset offset+n, l1 l1+l2 这两个区间
if offset+n <= l1:
    print offset,offset+n,0,0
elif offset<=l1  and offset+n<=l1+l2:
    print offset,l1,0,n-(l1-offset)
elif offset > l1 and offset + n > l1 + l2:
    print l1, l1, offset-l1, l2
elif offset > l1+l2:
    print l1, l1, l2, l2
# 还有完全包围的两种情况
elif offset<=l1 and offset+n>=l1+l2:
    print offset,l1,0,l2
elif offset > l1 and offset+n<l1+l2:
    print l1,l1,offset-l1,offset-l1+n

"""
华为19 1：对输入字符串检查是否有非法字符，输出合法字符（去重）和非法字符（不去重）
          对合法字符循环左移10次，再进行排序输出
          每输出一个字符串后用空格跟下一个字符串隔离，作为输出的所有的字符串之间只有一个空格
"""
# -*- coding:utf-8 -*-
# 读入数据
hefa = []
unhefa = []
while True:
    line = raw_input() # 一次只读一行
    if not line: # 如果是空行(^Z)就停止
        break
    # 检查此字符串是否非法：先去除空格，再使用isalnum函数
    a = ''.join(line.split())
    if a.isalnum():
        hefa.append(line)
    else:
        unhefa.append(line)
print 'hefa',hefa,'unhefa',unhefa
# 对合法字符串去重
after = []
for i in hefa:
    temp = i.split()
    temp = ''.join(temp) # 这两行去掉了字符串中的所有空格
    temp1 = list(set(temp))
    temp1.sort(key = temp.index) #使得顺序不乱
    temp2 = ''.join(temp1)
    after.append(temp2)
    print temp2, # 去重之后的字符串之间不再有空格
# 输出所有非法字符串
for i in unhefa:
    print i,
# 去重字符串循环左移十次
for i in after:
    for j in range(10):
        temp = i[:len(i)-1] + i[0] # 字符串专用的+：表示连接
        print temp,
        i = temp
# 去重字符串按ASCII字符表从小到大排列
after2 = sorted(after) # sorted函数就是按照ascii码排序的，按其他的排序可以调用sorted函数的key或cmp方法，异地排序
for i in after2:
    print i,

"""
华为19.2
填充字符串
"""
a = raw_input().split()
n = int(a[0])
final = []
for i in range(1,len(a)):
    if len(a[i])<8:
        temp0 = ['0']*(8-len(a[i]))
        temp1 = a[i] + ''.join(temp0)
        final.append(temp1)
    elif len(a[i])==8:
        final.append(a[i])
    elif len(a[i])>8:
        chushu = len(a[i])/8
        yushu = len(a[i])%8
        print 'yushu',yushu
        for j in range(chushu):
            final.append(a[i][(8*j):(8*j+8)])
        temp0 = ['0']*(8-yushu)
        print temp0
        temp1 = a[i][(chushu*8):] + ''.join(temp0)
        print temp1
        final.append(temp1)
final.sort()
for i in final:
    print i,

"""
华为19.3
abc3(ABC)
CBACBACBAcba
"""
""" 这种只能通过百分之20case，如示例所示的case"""
b = raw_input()
a = []
for i in b:
    a.append(i)
i = 0
final =[]
while i<len(a):
    j = i
    if a[i].isalpha():
        final.append(a[i])
    elif a[i].isdigit():
        start = i+2
        j = i+2
        while a[j] != ')' and a[j] != ']' and a[j] != '}':
            j += 1
        temp = ''.join(a[(i+2):j]) * int(a[i])
        final.extend(temp)
    i = j+1
result = ''.join(final)
print result[::-1]

""" 完整写法 """
# -*- coding:utf-8 -*-
# 进行各类括号的匹配，记录每一个配对的括号的坐标：最内部的括号排在列表的最前面
# indexs中每个list存储了两个坐标，分别代表着前括号出现的坐标和后括号出现的坐标
def calculate_index(strings):
    indexs = []
    d = []
    for i in range(len(strings)):
        if strings[i] in ['(', '{', '[']:
            d.append(i)
        elif strings[i] in [')', ']', '}']:
            indexs.append([d[-1], i])
            d.pop()
    return indexs
def replace(strs):
    indexs = calculate_index(strs)
    while len(indexs) > 0:
        index = indexs[0] # 仅仅取出此时最内部的括号（内部的第一个括号），如( )
        strs = strs[:index[0] - 1] + strs[index[0] + 1:index[1]] * int(strs[index[0] - 1]) + strs[index[1] + 1:]
        #        括号前的字符串    +      括号内的字符串         *     括号前的数字        +     括号后的字符串
        # 每一次的此行操作仅仅对此时最内部括号进行重复，并去掉括号
        indexs = calculate_index(strs) # 取得内部的第二个括号，如[ ]
    strs=strs[::-1] # 倒序输出
    print(strs)
strings=raw_input()
replace(strings)

"""
华为19.4
深度优先遍历DFS：找到从起点到达终点的路径个数。
即两点间的所有路径。
"""
line = map(int,raw_input().split())
n = line[0]
m = line[1]
state= []
for i in range(n):
    line = map(int,raw_input().split())
    state.append(line)
line = map(int,raw_input().split())
start_x = line[0]
start_y = line[1]
end_x = line[2]
end_y = line[3]
global count
count = 0
# 创建一个与海拔state同等大小的list，称为test
test = []
for i in range(n):
    test.append([0]*m)
def dfs(i,j,end_x,end_y):
    if i==end_x and j==end_y: # 有一条路径到达了终点，count+1
        count += 1
        return # 立即返回，终点始终不会标记为1
    test[i][j]=1 # 走过的格子标记为1
    for r,c in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
        if r>=0 and r< n and c>=0 and c< m and test[r][c]==0 and state[r][c]>state[i][j]:
            dfs(r,c)
            test[r][c]=0
            # 能执行到这一句，说明是成功了并return回来，将其置为1，下次可以继续到达这个点，不过是通过其他不同的路径
            # 下一次循环是从它的上一个点开始尝试，尝试in中的下一个可能点，但从这个点可以到达我们上回置为0的点了
            # 不好理解的话，画个图瞧一瞧嘛，终于搞明白了
            # https://www.cnblogs.com/Diliiiii/p/10305769.html
            # 这样始终不会产生完全相同的一条路径
dfs(start_x,start_y,end_x,end_y)
print count

# -*- coding:utf-8 -*-
"""
某厂19：给定一个整数数组A，拥有n个不重复的整数，找出数组中两个数之和出现最多的和
8
1 8 3 11 4 9 2 7
输出：11 10 12
"""
n =input()
a = map(int,raw_input().split())
dict1 = dict()
for i in range(len(a)):
    j = i+1
    while j>i and j<len(a):
        temp = a[i]+a[j]
        j += 1
        if str(temp) in dict1:
            dict1[str(temp)] += 1
        else:
            dict1[str(temp)] = 1
        temp = 0
# 找出最大的value值，然后对比每个value看是否都为最大值，若为，则输出这个key
max_value = max(list(dict1.values()))
print max_value
# s = sorted(dict1.items(),key=lambda x:x[1],reverse=True)
for k,v in dict1.items():
    if v==max_value:
        print k

"""
招银网络科技18.1：
我们有很多区域，每个区域都是从a到b的闭区间，现在我们要从每个区间中挑选至少2个点，那么最少可以挑选多少个点？
第一行是N（N<10000）,表示有N个区间，之间可以重复
然后每一行是ai,bi，持续N行，表示现在区间。均小于100000。
"""
"""
贪心，按右端点排序(左端点一样可以)，如果每个区间至少一个点，那么这个点一定是在右端点，
这样子可以让后续的区间更容易覆盖到这个点，从而减少选点的数量；
同理，如果每个区间至少两个点，那个这个两个点一定也在右端点和右端点前一个点。
"""

n = input()
temp = []
sol =[]
for i in range(n):
    l = list(map(int,raw_input().split()))
    temp.append(l[::-1])  # 倒着放入temp中之后，之后的sort函数默认使用区间的右端点进行了排序
temp.sort() # temp中存储着 [2,0],[4,2],[6,3],[7,4] 等坐标区域
sol.append(temp[0][0]-1) # 右端点前一个点
sol.append(temp[0][0]) # 右端点
for i in range(n-1): # 考虑前面n-1个区间
    # temp[i+1][-1]是第i+1个区域的左端点，sol[-1]是此时取到的之前区域的右端点，sol[-2]是之前区域中的右端点的前一个点
    # 若下一个区域的左端点<此时区域的右端点，大于右端点的前一个点，那么：仅将下一个区域的右端点加入sol中
    # 因为此时的sol[-2]依然属于此区域，没必要再添加一个
    if sol[-1]>=temp[i+1][-1] and sol[-2]<temp[i+1][-1]:
        sol.append(temp[i+1][0])
    # 若下一个区域的左端点>此时区域的右端点,即它们没有重复的部分，那么就要把这它的右端点以及右端点之前的一个点都加入其中
    elif sol[-1]<temp[i+1][-1]:
        sol.append(temp[i+1][0]-1)
        sol.append(temp[i+1][0])
print(len(sol))

"""
招银网络科技18.2：
考虑你从家出发步行去往一处目的地，该目的地恰好离你整数单位步长（大于等于1）。
你只能朝向该目的地或者背向该目的地行走，而你行走的必须为单位步长的整数倍，且要求你第N次行走必须走N步。
请就给出目的地离你距离，判断你是否可以在有限步内到达该目的地。如果可以到达的话，请计算到达目的地的最短总步数(不能到达则输出-1)。
距离目的地2， 需要3步：朝向走1，背向走2，朝向走3
"""
"""
采取暴力dfs的方法
"""

# -*- coding: utf-8 -*-
def find(x, val, distanceSet): # x表示本次走的步数（也就是最终到终点时的步数），distanceSet队列存储着 距起点的距离
    # 分两种情况考虑：x==0的时候讲0作为队列初始值添加到队列中，其实也可以直接将队列定为初始值0，只不过可以考虑一种
    # 起点即终点的特殊情况，直接返回x。
    if x == 0:
        distanceSet.add(x)
        if val in distanceSet:
            return x
        else:
            return find(x + 1, val, distanceSet)
    #         return 0
    elif x > 0:
        dSet = set() # 每次新的递归时将set设为0，相当于移除了之前队列中的元素
        for d in distanceSet: # 将d+x,d-x都存储到队列中，每一次递归，队列的长度就增长一倍
            dSet.add(d + x)
            dSet.add(d - x)
        if val in dSet:
            return x  # x表示本次走的步数（也就是最终到终点时的步数）
        else:
            return find(x + 1, val, dSet)
val = input()
distanceSet = set()
print(find(0, val, distanceSet))

"""
招银网络科技18.3：
我们部门要排队唱歌，大家乱哄哄的挤在一起，现在需要按从低到高的顺序拍成一列，但每次只能交换相邻的两位，请问最少要交换多少次
第一行是N（N<50000）,表示有N个人
然后每一行是人的身高Hi（Hi<2000000,不要怀疑，我们以微米计数），持续N行，表示现在排列的队伍
输出一个数，代表交换次数。
"""
# 比较直观的方法：交错冒泡法直接计算交换的次数，复杂度为O（n*n），太高了，不能通过所有的case
# -*- coding:utf-8 -*-
n = input()
a = []
for i in range(n):
    temp = input()
    a.append(temp)
# 采取交错排序的方式进行排序，每交换一次数据，count += 1
count = 0
def stagger_sort(lst):
    '''
    #交错冒泡排序
    #具体做法是 第一遍从左向右移动，下一遍从右向左移动，交替进行。
    #目的 解决冒泡排序中存在一些距离最终位置很远的元素导致对算法的拖累。
    '''
    global count # 注意global添加的位置
    e = 0 #奇偶 奇数正排序，偶数倒排序
    to_right = 1
    to_left = len(lst)-1 # 记录正序和倒叙开始排序位置，避免重复比对
    no_finish = True #循环标记符
    while no_finish:
        found = False #标记是否有元素移动，
        if e % 2 == 0:
            for j in range(to_right, len(lst)):
                #从左向右
                if lst[j-1] > lst[j]:
                    lst[j], lst[j-1] = lst[j-1], lst[j]
                    found = True
                    count += 1
            to_left -= 1
        else:
            for j in range(to_left, -1, -1):
                #从右向左
                if lst[j] > lst[j+1]:
                    lst[j], lst[j+1] = lst[j+1], lst[j]
                    found = True
                    count += 1
            to_right +=1
        if not found: #False 代表循环后没有发生过元素移动 意味着排完可以退出总循环
            no_finish = False
        e += 1
stagger_sort(a)
print count

# 方法二：用归并排序的方法查找逆序对
# -*- coding:utf-8 -*-
n = input()
data = []
for i in range(n):
    temp = input()
    data.append(temp)
def InversePairsCore( data, copy, start, end):
    if start == end:
        copy[start] = data[start]
        return 0
    length = (end - start) // 2
    left = InversePairsCore(copy, data, start, start + length)
    right = InversePairsCore(copy, data, start + length + 1, end)
    # i初始化为前半段最后一个数字的下标
    i = start + length
    # j初始化为后半段最后一个数字的下标
    j = end
    indexCopy = end
    count = 0
    # 对两个数组进行对比取值的过程
    while i >= start and j >= start + length + 1:
        if data[i] > data[j]:
            copy[indexCopy] = data[i]
            indexCopy -= 1
            i -= 1
            count += j - start - length
        else:
            copy[indexCopy] = data[j]
            indexCopy -= 1
            j -= 1
    # 剩下的一个数组未取完的操作
    while i >= start:
        copy[indexCopy] = data[i]
        indexCopy -= 1
        i -= 1
    while j >= start + length + 1:
        copy[indexCopy] = data[j]
        indexCopy -= 1
        j -= 1
    return left + right + count
def InversePairs(data):
    # write code here
    length = len(data)
    if data == None or length <= 0:
        return 0
    copy = [0] * length
    for i in range(length):
        copy[i] = data[i]

    count = InversePairsCore(data, copy, 0, length - 1)
    return count
print InversePairs(data)

"""
招银网络科技18.4：
这个游戏有三个因素：N，K，W
游戏开始的时候小招喵有0点，之后如果发现自己手上的点不足K点，就随机从1到W的整数中抽取一个（包含1和W），抽到每个数字的概率相同。
重复上述过程，直到小招喵获得了K或者大于K点，就停止获取新的点，这时候小招喵手上的点小于等于N的概率是多少？

输入：N = 6， K = 1， W = 10
输出：0.60000
说明：开始有0点，不足1点，从[1,10]中随机取一个整数（一共10个数字，所以每个数字取到的概率都是1/10），
获得后有6/10的概率小于6点，且满足大于1点的条件，概率为0.6
输出为概率值，保留5位小数
"""
"""
f(x)：当我们已经取出x个点时，小招喵手上的点小于等于N的概率
f(x)= 1/w ∗( f(x+1)+f(x+2)+...+f(x+W) )
dp[x] = the answer when Alice has x points

"""

n, k, w = list(map(int, input().split()))
dp = [0]*(k+w) #k-1能达到的最大值为k+w-1
for i in range(k, n+1):
    dp[i] = 1
s = min(w, n-k+1)
for i in range(k-1, -1, -1):
    dp[i] = s / w
    s += dp[i] - dp[i+w]
print(round(dp[0], 5))

class Solution(object):
    def new21Game(self, N, K, W):
        dp = [0.0] * (N + W + 1)
        # dp[x] = the answer when Alice has x points
        for k in xrange(K, N + 1): # 当取出的点数介于k到n之间时，输出1.0
            dp[k] = 1.0

        S = min(N - K + 1, W)
        # S = dp[k+1] + dp[k+2] + ... + dp[k+W]
        for k in xrange(K - 1, -1, -1):
            dp[k] = S / float(W)
            S += dp[k] - dp[k + W]

        return dp[0]
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 39、数组中出现次数超过一半的数字
"""
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
"""
import collections
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        tmp = collections.Counter(numbers) # 返回一个dict
        x = len(numbers)/2
        for k,v in tmp.items():
            if v>x:
                return k
        return 0

# 40、最小的k个数
"""
输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4.（注意k>n的情况,则输出[] ）
先排序，再取 tinput[:k].
"""
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if tinput is None:
            return
        if k > len(tinput):
            return []
        tinput = sorted(tinput)
        return tinput[:k]

# 41、数据流中的中位数
"""
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
"""
# -*- coding:utf-8 -*-
class Solution:
    # 首先，给这个类加一个初始化的程序，定义一个data,先用insert将num传到data中，然后对data进行各种操作
    def __init__(self):
        self.data = []

    def Insert(self, num):
        # write code here
        self.data.append(num)
        self.data.sort()

    def GetMedian(self,data):  # 记得将data作为一个参数传入！！！
        # write code here
        length = len(self.data)
        if length%2 == 0:
            return (self.data[length//2] + self.data[length//2-1])/2.0
            # // 整数除法,返回不大于结果的一个最大的整数./是浮点数除法
        else:
            return self.data[length//2]

# 42、连续子数组的最大和
"""
例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。
给一个数组，返回它的最大连续子序列的和.(子向量的长度至少是1)
注意：可以从中间开始连续！
"""
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if array is None:
            return
        max = 0
        sum = 0
        max1 = array[0] # 若array全为负数，则返回array中一个最大值
        for i in range(len(array)):
            if array[i] > max1:
                max1 = array[i]
            sum += array[i]
            if sum < 0:
                sum = 0
                continue
            if sum > max:
                max = sum
        if max > 0:
            return max
        else:
            return max1

# 43、1-n整数中1出现的次数
"""
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。
ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
"""
# -*- coding:utf-8 -*-
import collections
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        L = list()
        for i in range(1,n+1):
            L.extend(map(int,str(i)))
        tmp = collections.Counter(L)  # 返回一个dict
        return tmp[1]

# 44、数字序列中某一位的数字
"""
数字以0123456789101112131415...的格式序列化到一个字符序列中，请写一个函数，求任意第n位对应的数字。
"""

# -*- coding:utf-8 -*-
class Solution:
    def Findn(self, n):
        # write code here
        sum = 0
        for i in range(n):  # 实际上绝大多数情况都不需要遍历到i=n
            sum += len(str(i))
            if sum >= n: # 说明i的某一位就是我们要寻找的第n位对应的数字
                # 如果刚好等于n，则就是i的第一位，即i[0],即i[sum-n].注意，i此时为int，要先转为str
                return str(i)[sum-n]

# 45、把数组排成最小的数
"""
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。最终的输出应该是字符串。
"""
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers:
            return ""
        # 定义比较规则
        lmb = lambda n1,n2 : int( str(n1)+str(n2) ) - int( str(n2)+str(n1) )
        array = sorted(numbers,cmp=lmb)
        # 传入的numbers可迭代，按照匿名函数lmb形式（拼接起来的大小）来进行排列（增序）
        # 此时得到的array便是拼接起来最小到最大的，按顺序相连即可
        return ''.join([str(i) for i in array])

"""
# 注：sorted返回副本（sort在本地进行排序），cmp参数用于比较的函数
(1)按照元素长度排序
L = [{1:5,3:4},{1:3,6:3},{1:1,2:4,5:6},{1:9}]
def f(x):
    return len(x)
sort(key=f)
print L
[{1: 9}, {1: 5, 3: 4}, {1: 3, 6: 3}, {1: 1, 2: 4, 5: 6}]
(2)按照每个字典元素里面key为1的元素的值排序
L = [{1:5,3:4},{1:3,6:3},{1:1,2:4,5:6},{1:9}]
def f2(a,b):
    return a[1]-b[1]
L.sort(cmp=f2)
print L
"""

# 46、把数字翻译为字符串（没看）
"""
给定一个数字，按照如下规则翻译成字符串：0翻译成“a”，1翻译成“b”...25翻译成“z”。一个数字有多种翻译可能，
例如12258一共有5种，分别是bccfi，bwfi，bczi，mcfi，mzi。实现一个函数，用来计算一个数字有多少种不同的翻译方法。

c++版见此： https://www.jianshu.com/p/80e1841909b7   https://blog.csdn.net/xy_cpp/article/details/79000901

public class P231_TranslateNumbersToStrings {
    public static int getTranslationCount(int number){
        if(number<0)
            return 0;
        if(number==1)
            return 1;
        return getTranslationCount(Integer.toString(number));
    }
    //动态规划，从右到左计算。
    //f(r-2) = f(r-1)+g(r-2,r-1)*f(r);
    //如果r-2，r-1能够翻译成字符，则g(r-2,r-1)=1，否则为0
    public static int getTranslationCount(String number) {
        int f1 = 0,f2 = 1,g = 0;
        int temp;
        for(int i=number.length()-2;i>=0;i--){
            if(Integer.parseInt(number.charAt(i)+""+number.charAt(i+1))<26)
                g = 1;
            else
                g = 0;
            temp = f2;
            f2 = f2+g*f1;
            f1 = temp;
        }
        return f2;
    }
    public static void main(String[] args){
        System.out.println(getTranslationCount(-10));  //0
        System.out.println(getTranslationCount(1234));  //3
        System.out.println(getTranslationCount(12258)); //5
    }
}

"""

# 47、礼物的最大价值
"""
在一个 m*n 的棋盘中的每一个格都放一个礼物，每个礼物都有一定的价值（价值大于0）.你可以从棋盘的左上角开始拿各种里的礼物，
并每次向左或者向下移动一格，直到到达棋盘的右下角。给定一个棋盘及上面个的礼物，请计算你最多能拿走多少价值的礼物？
"""
"""
在动态规划求解这个问题的时候，我们找出到达每一行中每个位置的最大值，只与它上面和前面的值有关，不断更新
和算法图解中的背包问题不同，这里给出了每个格子的物品value，只需要一个与原始矩阵列数相等的一维向量来存储中间计算的值即可
https://blog.csdn.net/dugudaibo/article/details/79678890
背包问题需要自己画格子，纵坐标为每行能新加入的物品，列的间距为物品重量的最小公倍数，用P142的公式计算！！需要维护2行4列的矩阵！
再比如旅行行程的优化P147，但是仅适用于每个子问题都是离散的情况。最长公共子串P152，最长公共子序列P155
"""
# -*- coding:utf-8 -*-
class Solution:
    def getmaxValue(self, values, rows, cols):
        # write code here
        if not values or rows<=0 or cols<=0:
            return 0
        temp = [0] * cols  # 用于存放中间数值的临时数组
        for i in range(rows):
            for j in range(cols):
                left = 0
                up = 0
                if i>0:
                    up = temp[j]  # 上面框框的值就是此时j列的值，此时尚未更新
                if j>0:
                    left = temp[j-1] # 左边框框的值就是j-1列的值，已经更新
                temp[j] = max(up,left) + values[i*rows+j]    # 更新temp[j]
        return temp[-1]

# 48、最长不含重复字符的子字符串
"""
给定一个字符串，请找出其中无重复字符的最长子字符串。
例如，在”abcabcbb”中，其无重复字符的最长子字符串是”abc”，其长度为 3。 
对于，”bbbbb”，其无重复字符的最长子字符串为”b”，长度为1。
"""
"""
思路：遍历字符串中的每一个元素。借助一个dict来存储某个元素最后一次出现的下标。
用一个整形变量存储当前无重复字符的子串开始的下标。
https://blog.csdn.net/yurenguowang/article/details/77839381
"""
class Solution:
    def lengthOfLongestSubstring(self, s):
        # write your code here
        if s is None or len(s) == 0:
            return 0
        d = {}
        start = 0
        tmp = 0 # 记录此时序列的长度
        maxl = 0
        for i in range(len(s)):
            if s[i] in d and d[s[i]] >= start: # 若从某start开始的子串中出现了重复字符
                start = d[s[i]] + 1   # 新的start位置定为重复字符(两个中的前一个)的下一位,相当于把之前那个被重复的字符移出去
            tmp = i - start + 1 # 此时序列的长度
            d[s[i]] = i # 记录下此元素本次出现的坐标
            maxl = max(maxl,tmp)
        return maxl

# 49、丑数
"""
把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。
 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
"""
"""
丑数：如果一个数%2==0，就连续除以2；%3==0，就连续除以3；%5==0，就连续除以5。若最后得到的是1，那么这个数就是丑数。
逐一遍历时间消耗太大。试着找到一种只计算丑数的方法。丑数应该是另一个丑数乘以2、3或5的结果。
因此可以创建一个数组，里面的数字是排序好的丑数。观察 1 1*2 1*3 2*2 1*5 3*2  
"""
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if (index <= 0):
            return 0
        uglylist = [1]
        # index2:肯定存在某个丑数，排在它之前的每个丑数乘以2得到的结果都会小于已有的最大丑数，在它之后的每个丑数乘2的结果又太大
        index2 = 0
        index3 = 0
        index5 = 0
        for i in range(index-1):
            newUgly = min(uglylist[index2]*2 ,uglylist[index3]*3 ,uglylist[index5]*5)
            uglylist.append(newUgly)
            if (newUgly % 2 == 0):
                index2 += 1
            if (newUgly % 3 == 0):
                index3 += 1
            if (newUgly % 5 == 0):
                index5 += 1
        return uglylist[-1]

# 50、第一个只出现一次的字符
"""
在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
# 可以利用一个dict（key为位置，value为该位置的字符），若遍历到某字符 in d，则value+1，否则，增加一个新key，并将其value设为1
# 在d中第二次遍历，扫到的第一个value为1的
"""

import collections
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        d = collections.OrderedDict()
        sum = 0
        for i in range(len(s)):
            if s[i] in d:
                d[s[i]] += 1
            else:
                d[s[i]] = 1
        for i in range(len(s)):
            if d[s[i]] == 1:
                return i
        return -1
# 使用OrderedDict会根据放入元素的先后顺序进行排序。所以输出的值是排好序的。
# OrderedDict对象的字典对象，如果其顺序不同那么Python也会把他们当做是两个不同的对象
"""
拓展：字符流中第一个出现一次的字符：从字符流中只读出前两个字符go，第一个出现的字符是g，读前六个，google，是l。
如果当前字符流没有存在出现一次的字符，返回#字符。
思路：这是字符流，读出前两个时，输入的就是go，读出前六个时，输入的就是google
"""
# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.s=''
        self.dict1={} # 保存读入的字符流中每个元素的个数
    def FirstAppearingOnce(self):
        # write code here
        for i in self.s:
            if self.dict1[i]==1:
                return i
        return '#'
    def Insert(self, char): # 字符流计数程序，读入一个char，则添加到s中，并对这个字符的数目+1，若不存在则添加这个key
        # write code here
        self.s=self.s+char
        if char in self.dict1:
            self.dict1[char]=self.dict1[char]+1
        else:
            self.dict1[char]=1
# 51、数组中的逆序对  逐一比较某一个数和它后面的所有数的复杂度为O(n*n)，优化方法见 P268
"""
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
题目保证输入的数组中没有的相同的数字
数据范围：
	对于%50的数据,size<=10^4
	对于%75的数据,size<=10^5
	对于%100的数据,size<=2*10^5
"""
# python代码
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        return self.inverseCount(data[:], 0, len(data) - 1, data[:]) % 1000000007

    def inverseCount(self, tmp, start, end, data):
        if end - start < 1:
            return 0
        if end - start == 1:
            if data[start] <= data[end]:
                return 0
            else:
                tmp[start], tmp[end] = data[end], data[start]
                return 1
        mid = (start + end) // 2
        cnt = self.inverseCount(data, start, mid, tmp) + self.inverseCount(data, mid + 1, end, tmp)
        # print(start, mid, end, cnt, data)
        i = start
        j = mid + 1
        ind = start

        while (i <= mid and j <= end):
            if data[i] <= data[j]:
                tmp[ind] = data[i]
                i += 1
            else:
                tmp[ind] = data[j]
                cnt += mid - i + 1
                j += 1
            ind += 1
        while (i <= mid):
            tmp[ind] = data[i]
            i += 1
            ind += 1
        while (j <= end):
            tmp[ind] = data[j]
            j += 1
            ind += 1
        return cnt
# c++
"""
static const auto io_sync_off = []() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    return nullptr;
}( );
class Solution {
public:
    static constexpr int P = 1000000007;
    vector<int>::iterator it;
    int InversePairs(vector<int>& data) {
        it = data.begin();
        if (data.empty())return 0;
        vector<int> dup(data);
        return merge_sort(data.begin(), data.end(), dup.begin());
    }
    //template<class RanIt>
    using RanIt = vector<int>::iterator;
    int merge_sort(const RanIt& begin1, const RanIt& end1, const RanIt& begin2) {
        int len = end1 - begin1;
        if (len < 2)return 0;
        int mid = ( len + 1 ) >> 1;
        auto m1 = begin1 + mid, m2 = begin2 + mid;
        auto i = m1, j = end1, k = begin2 + len;
        int ans = ( merge_sort(begin2, m2, begin1) + merge_sort(m2, k, m1) ) % P;
        for (--i, --j, --k; i >= begin1 && j >= m1; --k) {
            if (*i > *j) {
                *k = *i, --i;
                ( ans += j - m1 + 1 ) %= P;
            } else *k = *j, --j;
        }
        if (i >= begin1)copy(begin1, i + 1, begin2);
        else copy(m1, j + 1, begin2);
        return ans;
    } 
};
"""

# 52、两个链表的第一个公共节点
"""
输入两个链表，找出它们的第一个公共结点。
"""
"""
链表有公共结点，意味着这两个链表从某一节点开始，他们的.next都指向同一个节点，由于是同一个节点，后面所有的节点也必然相同
所以公共节点出现在链表的尾部部分，可以先把一链表的所有节点放到一个list中，然后遍历第二个链表看看有没有节点在list1中。
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        list1 = []
        node1 = pHead1
        node2 = pHead2
        while node1:
            list1.append(node1.val)
            node1 = node1.next
        while node2:
            if node2.val in list1:
                return node2
            else:
                node2 = node2.next
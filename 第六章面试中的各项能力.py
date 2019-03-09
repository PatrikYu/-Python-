import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 53、在排序数组中查找数字
"""
统计一个数字在排序数组中出现的次数。
"""
# 思路：既然是排序数组，直接用二分法啊    start = 0  end = len(data)-1
# 当找到这个数字k之后，查看上一个数字，若不是k，说明这是第一个k，若是k则进入递归，直到找到第一个k的位置，
# 同理可寻最后一个k的位置
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        return data.count(k)
"""
在范围0~n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字
直观思路：求解这n个数字的和-这n-1个数字的和，它们的差就是缺少的那个数字
利用递增：假设m不在数组中，那么m+1就处在下标为m的位置，可以发现：m正是数组中第一个值和下标不相等的元素，二分法实现：
if data[mid]==mid: start = mid+1
elif data[mid-1]==mid: return mid
else: end = mid -1
"""

"""
单调递增数组里的每个元素都是整数且唯一，找出数组中任意一个数值等于其下标的元素
直观：逐一遍历
二分法：if data[mid]==i : return
if data[i]>i : end = mid-1
else : start=mid+1
"""

# 54、二叉搜索树的第K大节点
"""
给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。
"""
# 按照中序遍历的顺序遍历一棵二叉搜索树，则遍历序列的数值是递增排序的，然后直接取出索引为[k-1]的值即可
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        global result   # 定义一个全局变量用来存储中序遍历之后递增的序列
        result = []
        self.midnode(pRoot)  # 调用中序遍历函数
        if k<=0 or len(result)<k:
            return None
        else:
            return result[k-1]

    def midnode(self,root):
        if not root:
            return None
        self.midnode(root.left)
        result.append(root)
        self.midnode(root.right)

# 55、二叉树的深度
"""
输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
"""
"""
思路：如果一棵树只有一个节点，那么它深度为1.若根节点只有左子树，那么树的深度就是左子树深度+1，右子树同理.
若左右都有，那么树的深度就是 max(左或右子树的深度)，用递归的方法实现
"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        return self.recur(pRoot,1)

    def recur(self,root,level):
        maxlevel = level
        if root.left != None:
           maxlevel = max(maxlevel,self.recur(root.left,level+1))
        if root.right != None:
            maxlevel = max(maxlevel,self.recur(root.right,level+1))
        return maxlevel
"""
输入一棵树的根节点，判断该树是不是平衡二叉树（二叉树左右子树的深度相差不超过1）
思路：调用上面所写的返回二叉树深度的函数
"""
    def isbalanced(self,pRoot):
        if not pRoot:
            return True
        left_length = self.TreeDepth(pRoot.left)
        right_length = self.TreeDepth(pRoot.right)
        if abs(left_length-right_length)>1:
            return False
        return self.isbalanced(pRoot.left) and self.isbalanced(pRoot.right)
# 为了避免重复访问节点，可以采取后序遍历的方式

class Solution:
    __is_balanced = True  # 放在函数外部，相当于一个全局变量
    def getDepth(self, pRoot):
        if pRoot == None:
            return 0
        left_depth = self.getDepth(pRoot.left)
        right_depth = self.getDepth(pRoot.right)
        if abs(left_depth-right_depth) > 1:
            self.__is_balanced = False  # 一旦任意一个子树出现这种情况，立即置于false，后面的语句不会继续执行，只会一步步返回
        return left_depth+1 if left_depth>right_depth else right_depth+1 # 返回此时子树深度
    def IsBalanced_Solution(self, pRoot):
        self.getDepth(pRoot)
        return self.__is_balanced


# 56、数组中数字出现的个数
"""
数组中只出现一次的数字：一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个只出现一次的数字。
"""
# 思路：可以建立一个新的list，第一次遇到该字符则append进去，第二次遇到则remove，由于出现了偶数次，最后一定被移出去

# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        tmp = list()
        for i in array:
            if i in tmp:
                tmp.remove(i)
            else:
                tmp.append(i)
        return tmp

"""
只有一个数字出现了一次，其他数字都出现了三次，请找出此数字
"""
import collections
# -*- coding:utf-8 -*-
class Solution:
    def FindNumsAppearOnce(self, array):
        # write code here
        tmp = collections.Counter(array)  # 返回一个dict
        for k,v in tmp.items():
            if v==1:
                return k

# 57、和为s的数字
"""
在一个递增的数组中，给定s，在数组中找到两个数，使得它们的和为s。
思路：start=a[0] end=a[len(a)-1] if start+end<s:start=start+1 else: end=end-1 while start<end 这样就做到了复杂度为O（n）
如果有多对数字的和等于S，输出两个数的乘积最小：
每队都存储起来，然后最后计算哪个最小。（理论上按我们这种方法最小的应该是第一组，下面代码就直接输出了）
"""
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        if not array:
            return []
        i = 0   # start
        j = len(array) - 1 #end
        while i<j:
            if i == j:
                return []
            if array[i] + array[j] == tsum:
                return [array[i], array[j]]
            elif array[i] + array[j] > tsum:
                j -= 1
            else:
                i += 1
        return []

"""
输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
如S=15,1-5,4-6,7-8
思路：同样定义start和end，只不过初始化的值start=a[0],end=a[1],以s=9为例走一遍，小了就移动end向右一步，大了就移动start向左一步
注意等于的时候下一步移动start、end均可。循环进行的条件：while small < (tsum+1)//2，否则光是start+end就大于tsum了
"""
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        if tsum<3:
            return []
        start = 1
        end = 2
        sum = start + end # 序列所有值累加到这，没必要写一个子函数，for循环
        output = list()
        while start < (tsum+1)//2:
            if sum==tsum:
                output.append(range(start,end+1))
                end += 1
                sum += end
            elif sum < tsum:
                end += 1
                sum += end
            else:
                sum -= start
                start += 1
        return output

# 58、翻转字符串
"""
翻转单词顺序：输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。student. a am I
"""
# 思路：第一步翻转句子中所有的字符，第二步再翻转每个单词中字符的顺序，通过扫描空格来确定每个单词的起始和终止位置
# python:首先通过空格切割字符串，每一个单词作为list中一项，join函数将list中的每个单词用空格连接起来，并且仅仅对list中每个元素做倒序
"""
['what', 'the', 'fuck.']
fuck. the what
"""
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        l=s.split(' ')  # 以空格来分割单词
        return ' '.join(l[::-1])

"""
左旋转字符串：对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
思路：和上面的例题有关，可以把原字符串当做两部分，直接切开然后拼接起来就好
"""
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        return s[n:]+s[:n]

# 59、队列的最大值
"""
给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}
"""
# 思路:滑动窗口可以看成一个队列，当窗口滑动时，处于窗口的第一个数字被删除，同时在窗口末尾添加一个新的数字
# 之前题30用O(1)时间得到最小值的栈，题9讨论了如何用两个栈实现一个队列。这样将时间复杂度降低到了O(n)

# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if size <= 0:
            return []
        res = []
        for i in xrange(0,len(num)-size+1): # 每次窗口滑动时候停留的第一个索引值
            res.append(max(num[i:i+size]))
        return res

"""
定义一个队列并实现函数max得到队列里的最大值，要求函数max、push_back和pop_front的时间复杂度都是O(1)
#include<bits/stdc++.h>
using namespace std;

//带max函数的队列模板类 
template<typename T> class QueueWithMax {
	private:
		//这个自定义队列的元素结点结构体
		struct InternalData { 
			T number;//元素的值,泛型类型 
			int index;//插入时被赋予的下标 
		};
		deque<InternalData> data;//存队列实际内容的双端队列 
		deque<InternalData> maximums;//存最大值的双端队列 
		int currentIndex;//接下来要插入元素应该赋予的下标 

	public:
		//构造器 
		QueueWithMax() : currentIndex(0) {//初始化下标值为0 
		}

		//入队,相当于上一题里面滑动窗口滑入一个数字 
		void push_back(T number) {
			//和老数字比较,把双端队列里不比他大的老元素都删掉 
			while(!maximums.empty() && number >= maximums.back().number)
				maximums.pop_back();
 
			InternalData internalData = { number, currentIndex };//创建元素结点
			data.push_back(internalData);//进入存元素的队列 
			maximums.push_back(internalData);//也进入最大值队列 

			++currentIndex;//id自增1,用来给下一个插入的元素用 
		}

		//出队,相当于滑出窗口,不过这里可以视为窗口从左侧内陷1造成的 
		void pop_front() {
			if(maximums.empty())//队列为空不能出队 
				throw new exception();//queue is empty
			
			//这个就比上一题的滑出容易多了,直接判断是不是当前最大 
			if(maximums.front().index == data.front().index)
				maximums.pop_front();//如果是,就把那个当前最大也拿掉(从队头) 

			data.pop_front();//出队 
		}

		//获取最大值
		//尾接const表示成员函数隐含传入的this指针为const指针
		//即在这个成员函数内不能修改非mutable成员 
		T max() const {
			if(maximums.empty())//为空就没有最大值 
				throw new exception();//queue is empty
			//存最大值的双端队列的队头元素的number域 
			return maximums.front().number;
		}
};

int main() {
	QueueWithMax<int> queue;
	queue.push_back(2);
	cout<<queue.max()<<endl;//2
	queue.push_back(4);
	cout<<queue.max()<<endl;//4
	queue.push_back(3);
	cout<<queue.max()<<endl;//4
	queue.push_back(2);
	cout<<queue.max()<<endl;//4
	queue.pop_front();
	cout<<queue.max()<<endl;//4
	queue.pop_front();
	cout<<queue.max()<<endl;//3
	return 0;
}

"""

# 60、n个骰子的点数
"""
n个骰子的所有点数的排列数为6^n，需要先统计出每个点数出现的次数，再除以6^n。
思路一：
    基于递归求骰子点数，时间效率不够高。
先把骰子分成两堆，第一堆只有一个，第二堆有n-1个，
单独的那一个可能出现1到6的点数，我们需要计算从1-6的每一种点数和剩下的n-1个骰子来计算点数和。
还是把n-1个那部分分成两堆，上一轮的单独骰子点数和这一轮的单独骰子点数相加，然后再和剩下的n-2个骰子来计算点数和。
定义一个长度为6n-n+1的数组，和为s的点数出现的次数保存到数组的第s-n个元素里。
python中可以用一个dict来存储，每次点数的和在dict中新增或增1

	
#include <iostream>
#include <cstdio>

using namespace std;

int g_maxValue = 6;

void Probability(int original, int current, int sum, int *pProbabilities)
{
    if (current == 1)
    {
        pProbabilities[sum - original]++;
    }
    else
    {
        for (int i = 1; i <= g_maxValue; ++i)
        {
            Probability(original, current - 1, i + sum, pProbabilities);
        }
    }
}

void Probability(int number, int *pProbabilities)
{
    for (int i = 1; i <= g_maxValue; ++i)
    {
        Probability(number, number, i, pProbabilities);
    }
}

void PrintProbability(int number)
{
    if (number < 1)
    {
        return;
    }
    int maxSum = number * g_maxValue;
    int *pProbabilities = new int[maxSum - number + 1];
    for (int i = number; i <= maxSum; ++i)
    {
        pProbabilities[i - number] = 0;
    }

    Probability(number, pProbabilities);

    int total = pow( (double)g_maxValue, number);
    for (int i = number; i <= maxSum; ++i)
    {
        double ratio = (double)pProbabilities[i - number] / total;
        printf("%d: %e\n", i, ratio);
    }
    delete[] pProbabilities;
}

int main()
{
    PrintProbability(6);
    return 0;
}
"""


# 61、扑克牌中的顺子
"""
五张牌，若是连续的，则为顺子，A为1，J为11，Q为12，K为13，大王小王可以看成任意数字。
思路：可以把大王小王看作0。首先把数组排序；其次统计数组中0的个数；最后统计排序之后的数组中相邻数字之间的空缺总值
若空缺的总数小于或者等于0的个数，那么这个数组就是连续的。若非0数字重复出现，则该数组绝不是连续的
"""
# -*- coding:utf-8 -*-
import collections
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if not numbers:
            return False
        c = collections.Counter(numbers)
        m = c[0] # 统计0的个数（大小王）
        new_list = [i for i in numbers if i>0]
        new_list.sort()
        n = 0
        for j in range(len(new_list)-1):
            if new_list[j+1] - new_list[j] > 0:
                n += new_list[j+1] - new_list[j] # 统计空缺值
            else:
                return False # 若有相等元素，返回False
        if n <= m+len(new_list)-1:
            return True
        else:
            return False

# 62、圆圈中最后剩下的数字（约瑟夫环问题）
"""
0,1,...,n-1这n个数字排成一个圆圈，从0开始数，每次删除第m个数字（从目前存在的地方向前走m-1步，删除之）。
如 0,1,2,3,4。从0开始每次删除第三个数字，则删除的前4个数字依次为2,0,4,1，因此最后剩下的数字是3
两种解法：第一种构造一个环形链表，每次在这个链表中删除第m个节点
第二种：找规律,设被删去的位置为i，从i+1开始数，n变为n-1，i=0,m-1=2被删除的位置为 2%5=2，i=2,将i=2的位置删除
接下来使得被删除位置为0，即（2+i）%4=0。接下来删去3，位置为2,有（2+0）%3 =2
因此下次删除位置为 (m-1+i)%len(newlist)
"""
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if not n or not m:
            return -1
        res = range(n)
        i = 0
        while len(res)>1:
            i = (m-1+i)%len(res)
            res.pop(i)
        return res[0]

# 63、股票的最大利润
"""
一只股票在某些时间节点的价格为{9,11,8,5,7,12,16,14}。若在价格为5的时候买入并在价格为16的时候卖出，则能收获最大的利润11
寻找数组中拥有最大差值的一对数
"""
"""
暴力搜索太过复杂。可以先定义函数 diff(i)为当卖出价为数组中第i个数字时可能获得的最大利润。显然，在卖出价固定时，只要记住
前面i-1个数字时的最小值即可。
"""
# -*- coding:utf-8 -*-
class Solution:
    def maxbenefit(self, time):
        if not time or len(time)==1:
            return 0
        min = time[0]
        for i in range(1,len(time)-1):
            benefit = time[i]-min
            if time[i] < min:
                min = time[i]
        return benefit


# 64、求1+2+3+...+n
"""
求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
"""
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        def f(x,y):
            return x+y
        return reduce(f,range(n+1))
    # map是对序列中的每个元素执行前面的函数操作
    # reduce把一个函数（必须接受两个参数）作用在序列上，reduce把结果继续和序列的下一个元素做累积计算
    # like this : reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)

# 65、不用加减乘除做加法
"""
写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
"""
# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        # write code here
        s=[]
        s.append(num1)
        s.append(num2)
        return sum(s)

# 66、构建乘积数组
"""
给定一个数组A[0,1,...,n-1]（不是数值，是索引而已）,请构建一个数组B[0,1,...,n-1],
其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
思路：将它分为A[0]*...*A[i-1],以及A[i+1]*...*A[n-1]。注意：当A长度为1时即为0时，取B[0]=1。
"""
# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        # write code here
        if not A:
            return []
        # 计算前半部分
        num = len(A)
        B = [None]*num
        B[0] = 1
        for i in range(1,num):
            B[i] = B[i-1]*A[i-1]
        # 计算后面一部分
        # 自下而上,最后才对B[1]进行乘积，而且它是下半部分乘的最多的
        # 保留上次的计算结果乘本轮新的数,因为只是后半部分进行累加，所以设置一个tmp,能够保留上次结果
        tmp = 1
        for i in range(num-2,-1,-1): # 注意要从倒数第二项开始，因为B[n-1]项已经乘完了，一直乘到第二项A[1]
            tmp *= A[i+1]
            B[i] *= tmp
        return B

# 67、把字符串转换为整数
"""
将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，
要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。
"""

# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        # write code here
        try:
            return int(s)
        except Exception as e:  # 此语句可以捕获与程序退出 sys.exit()相关之外的所有异常
            return 0

# 68、树中两个节点的最低公共祖先
"""
是二叉树，并且是二叉搜索树：
思路：
二叉搜索树是经过排序的，位于左子树的节点都比父节点小，位于右子树的节点都比父节点大。
既然要找最低的公共祖先节点，我们可以从根节点开始进行比较。
若当前节点的值比两个节点的值都大，那么最低的祖先节点一定在当前节点的左子树中，则遍历当前节点的左子节点；
反之，若当前节点的值比两个节点的值都小，那么最低的祖先节点一定在当前节点的右子树中，则遍历当前节点的右子节点；
这样，直到找到一个节点，位于两个节点值的中间，则找到了最低的公共祖先节点。


/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null||root==p||root==q)
            return root;
        if(root.val>p.val&&root.val>q.val)
            return lowestCommonAncestor(root.left,p,q);
        if(root.val<p.val&&root.val<q.val)
            return lowestCommonAncestor(root.right,p,q);
        else
            return root;
    }
    
如果存在parent指针，则分别从输入的p节点和q节点指向root根节点，其实这就是两个单链表。问题转化为求两个单链表相交的第一个公共节点

那如果不存在parent指针呢？
用两个链表分别保存从根节点到输入的两个节点的路径，然后把问题转换成两个链表的最后公共节点。

public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null||root==p||root==q)
            return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if (left == null) 
            return right;
        if (right == null) 
            return left;
        return root;
    }

"""
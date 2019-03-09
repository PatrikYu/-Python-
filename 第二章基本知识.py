import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 3、数组中重复的数字
"""
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。
也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 
例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
但不能修改原来的数组。
"""
# coding:utf-8
import collections # 需要用到里面的Counter计数器
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self,numbers,duplication):
        flag = False
        c = collections.Counter(numbers) # 对输入的numbers序列进行计数，并生成一个dict
        for k,v in c.items():
            if v>1: # 计数量大于1
                duplication[0]=k
                flag = True
                break
        return flag

# 4、二维数组中的查找
"""
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数
"""

# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        rows = len(array)-1     # len(array)即array的行数
        cols = len(array[0])-1  # 取第一行array[0]，它的长度即为列数
        # 用i，j来表示此时取到数的坐标,首先取右上角
        i = 0
        j = cols
        while i<=rows and j>=0:
            if target < array[i][j]: # 那么target在左边区域，剔除此列
                j = j-1
            elif target > array[i][j]:
                i = i+1
            else:
                return True
        return False

# 5、字符串（替换空格）

"""
请实现一个函数，将一个字符串中的每个空格替换成“%20”。
例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
"""

# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        s = list(s) # 把原字符串转为list
        count = len(s)
        for i in range(0,count): # 创建一个0到count-1的list供迭代
            if s[i]==' ':
                s[i]='%20'
        return ''.join(s)

# 6、从尾到头打印链表
"""
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
"""
"""
单链表的结点是个二元组，elem保存元素值，链接域next保存下一个结点的标识
掌握一个链表，只需要用一个变量保持着这个表的首结点的引用（此变量称之为表头变量head或表头指针）
为了表示一个链表的结束，只需在表的最后结点设置一个空链接
链表操作 P80
"""
"""
题目给的class是以二元组定义的一个单链表
思路：要求返回一个list，先建立一个空list，将每个结点的val依次取出insert到list的0位置，这样就实现了倒序
"""

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        l = [] # 定义一个空列表
        head = listNode # 取得表头变量（第一个节点）
        while head:     # 链表不为空
            l.insert(0,head.val)
            head = head.next
        return l

# 7、重建二叉树

""""
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
"""
"""
只含有一个结点的二叉树是一棵单点树（只有根结点0），注意只有根结点的树高度为0。
对于二叉树，只有一个分支时，必须说明它是左分支还是右分支。
一个结点的子结点个数称为该结点的度数，分支结点的度数可以是1或2，叶结点的度数为0.
完全二叉树（最下一层可以不满，结点在左边连续排列，空位在右）的性质（判断有无左右叶结点）P180
这说明完全二叉树到线性结构有自然的双向映射，可以方便地从相应线性结构恢复完全二叉树
对于有n个结点的二叉树，其最长路径的平均值为O（log n）

遍历： （对每一个子二叉树）
前序遍历：根，左，右 
中序遍历：左，根，右 （又称为 对称序列）
后序遍历：左，右，根
定理：若知一棵二叉树的对称序列，又知另一遍历序列，就可唯一确定此二叉树
宽度优先遍历（层次顺序遍历(Breadth First Search)）：从上到下，从左到右，逐层往下数就ok了,利用队列
深度优先遍历(Depth First Search):先访问根结点，然后遍历左子树接着是遍历右子树，可以利用堆栈的先进后出的特点
"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin): # pre和tin为list
        # write code here
        if len(pre) == 0: # 空树
            return None
        if len(pre) == 1: # 单点树
            return TreeNode(pre[0])
        else:
            # 先序遍历重建代码：利用reConstructBinaryTree(前，中)
            val = pre[0]
            tree = TreeNode(val)    # 以树根root开始，不断添加其左右子树，pre[0]为根结点
            # 利用self.reConstructBinaryTree(先序的子树序列，中序的子树序列)
            # index() 函数用于从列表中找出某个值第一个匹配项的索引位置（list的索引从0开始）
            index = tin.index(val) # 这是 根结点在中序中的索引值
            tree.left = self.reConstructBinaryTree(pre[1 : index+1],tin[:index])
            tree.right = self.reConstructBinaryTree(pre[index+1:],tin[index+1:])

            # 后序遍历重建代码：利用buildTree(中，后)
            val = post.pop()
            tree = TreeNode(val)
            index = tin.index(val)
            tree.left = self.buildTree(tin[:index], post[:index])
            tree.right = self.buildTree(tin[index + 1:], post[index:])
        return tree

# 8、二叉树的下一个节点
"""
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。
注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
"""
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None   # 这就是结点指向父结点的指针
class Solution:
    def GetNext(self, pNode): # 输入pNode为一个结点
        # write code here
        if pNode == None:
            return None
        if pNode.right: # 若存在右子树,下一结点就是它的最左子结点
            left1 = pNode.right
            while left1.left: # 取得最左的子结点
                left1 = left1.left
            return left1
        p = pNode
        while pNode.next:
            temp = pNode.next #取得父结点，若pNode是父结点的左子结点，下一结点就是取得的父结点
            if temp.left == pNode:
                return temp
            pNode = temp  # 沿着父结点的指针向上遍历，直到找到那么一个左子结点或到达根结点（即pNode.next == None）

# 9、两个栈实现队列
"""
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
"""
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self): # 先自己在初始化里面定义两个栈 self.stack1,self.stack2
        self.stack1=[]
        self.stack2=[]
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self): # 当stack2空时，首先将stack1中的所有元素先push到Stack2中，再对stack2执行pop操作
        # return xx
        if self.stack2==[]:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
        # 非空时，直接pop即可
        return self.stack2.pop()

# 10、裴波那契数列
"""
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39
"""
"""
if 符合基线条件：不再递归，return
else：递归
"""
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return self.Fibonacci(n-1)+self.Fibonacci(n-2)
# 尴尬的事情发生了，程序运行超时，我还是乖乖写循环吧
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            s = []*(n+1) # 定义一个长度为n+1的list
            s.append(0)
            s.append(1)
            # range生成一个list,xrange生成一个生成器（生成一个取出一个，优化内存）
            for i in xrange(2,n+1): # 生成第三项到第n项
                s.append(s[i-1]+s[i-2])
            return s[n]

"""
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
"""
"""
设n级台阶有f(n)种跳法，第一次跳一步，此时跳法数目为f(n-1);第一次跳两步，此时跳法数目为f(n-2)。
故总的跳法数目为f(n-1)+f(n-2)。
n=1,f(1)=1
n=2,f(2)=2
n=3.f(3)=3  就是裴波那契数列，改一改就好
"""
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number == 0:
            return 0
        if number == 1:
            return 1
        if number == 2:
            return 2
        if number >= 3:
            s = []*(number+1) # 定义一个长度为n+1的list
            s.append(0)
            s.append(1)
            s.append(2)
            # range生成一个list,xrange生成一个生成器（生成一个取出一个，优化内存）
            for i in xrange(3,number+1): # 生成第4项到第n项
                s.append(s[i-1]+s[i-2])
            return s[number]
"""
用8个2*1的小矩形横着或竖着无重叠地覆盖2*8的矩形，总共有多少方法？
这也是一个裴波那契数列问题。
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？ 
思考： 2*1 1 种； 2*2 2种 2*3 3种 2*4 5种 
dp[n]=dp[n−1]+dp[n−2]dp[n]=dp[n−1]+dp[n−2]
"""
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        # write code here
        if number<=2:
            return number
        dp = [1,2]
        for i in range(number-2):
            dp.append(dp[-1]+dp[-2])
        return dp[-1]

"""
变态跳台阶：一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
数列：1 1 2 4 8 16 （是前面所有值的累加，恰好是 pow(2,number-1)  if number <= 0: return 0 ）
"""

# 11、旋转数组的最小数字
"""
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
"""
"""
旋转数组可以划分为两个排序的子数组，而且前面子数组的元素都>=后面子数组元素，而最小的元素恰好是这两个子数组的分界线。排序的数组
寻找最小值可以采用二分法，复杂度为O（log n）
"""
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        l=len(rotateArray)
        if l==0:
            return 0
        if rotateArray[0]<rotateArray[l-1]:
            return rotateArray[0]
        # 直接遍历吧（当第一个=最后一个=中间时，其他情况可以采用二分法），这种方法的速度为700ms（还好时间限制3秒）
        pre = -1
        for num in rotateArray:
            if num < pre:
                return num    # 由于两个子序列均递增，若出现一个比它小的，必为最小值
            pre = num
        return rotateArray[0]   # 如果大家都相等，就返回第一个值吧

# 12、矩阵中的路径（回溯法）
# 回溯法就只能用递归了呀（需要存储在栈里）
"""
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 
例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，
但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
"""
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path): # matrix是个一维数组(可能是list或其他)，假装摆成一个3*4的矩阵
        # write code here
        for i in range(rows):
            for j in range(cols):
                if matrix[i*cols+j] == path[0]:  # 选中一个路径起点
                    if self.find(list(matrix),rows,cols,path[1:],i,j):
                        return True
    def find(self,matrix,rows,cols,path,i,j): # 此时matrix是个list
        if not path:  # 如果已经为空，返回True
            return True
        matrix[i*cols+j] = '0' # 确保往回走一步时的if语句不成立，避免进入死循环！！！！！！
        # 接下来在上下左右四个方向寻找第二个字符（此时为path[0]）,某个方向是边界则不会寻找
        if j+1<cols and matrix[i*cols+j+1]==path[0]:
            return self.find(matrix, rows, cols, path[1:], i, j+1)
        elif j-1>=0 and matrix[i*cols+j-1]==path[0]:
            return self.find(matrix, rows, cols, path[1:], i, j - 1)
        elif i+1<rows and matrix[(i+1)*cols+j]==path[0]:
            return self.find(matrix, rows, cols, path[1:], i+1, j)
        elif i-1>=0 and matrix[(i-1)*cols+j]==path[0]:
            return self.find(matrix, rows, cols, path[1:], i-1, j)
        else:
            return False

# 13、机器人的运动范围

"""
地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，
但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。
但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
"""
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        # 设置一个dict全局变量，用来存储每个坐标的到达情况，能到达value设为1，不能到达设为0
        global dic   # 注意一定要设置这么一个全局变量！否则会报错：未设置全局变量dic
        dic = {}
        return self.count1(0,0,threshold,rows,cols)
    def count1(self,r,c,thresold,rows,cols):

        # 判断坐标是否超过边界(注意有0和最大值两个边界)
        if r<0 or r>=rows or c<0 or c>=cols:
            return 0
        if dic.get( (r,c) ):  # 若此坐标已被标记能到达，则return0，不执行下一句对四周探索的语句，避免死循环
            return 0
        # 判断是否符合阈值
        # map函数对字符串中的每一个字符进行操作，然后返回一个list,这种操作可以适应任何位数
        if sum(map(int,str(r)+str(c))) > thresold: # 之前报错，因为如果之前rc是个负数的话，int('-')报错，放到后面来
            return 0
        else:
            dic[ (r,c)] = 1
        # 前面加的这个一很关键，每递归一次，若后面四个返回的都不是0，在这个阶段，将会加四个1，代表这四个格子都能到达
        return 1+self.count1(r+1,c,thresold,rows,cols)+self.count1(r-1,c,thresold,rows,cols)\
               +self.count1(r,c+1,thresold,rows,cols)+self.count1(r,c-1,thresold,rows,cols)

# 14、动态规划与贪婪算法

# 贪婪算法（用最少的广播台覆盖所有州）
# 当遇到NP问题，列出所有的解复杂度太高，采用近似算法，贪婪算法每次选取的都是局部最优解，是近似的全局最优解

# 每次选出覆盖最多未覆盖州的广播台
# 首先，创建一个列表，其中包含要覆盖的州
states_needed = set(["mt","wa","or","id","nv","ut","ca","az"]) # 集合不能包含重复的元素
# 可供选择的广播台清单
stations = {}
stations["kone"] = set(["id","nv","ut"])
stations["ktwo"] = set(["wa","id","mt"])
stations["kthree"] = set(["or","nv","ca"])
stations["kfour"] = set(["nv","ut"])
stations["kfive"] = set(["ca","az"])
# 最后，需要使用一个集合来存储最终选择的广播台
final_stations = set()

while states_needed:
    best_station = None
    states_covered = set()
    for station,states in stations.items(): # station为kone,ktwo等等；states为取到的station所包含的车站set
        covered = states_needed & states # 目前选中的station与需要覆盖的州的重合的州
        if len(covered) > len(states_covered): # 选出此时覆盖州最多的广播台
            best_station = station
            states_covered = covered
    states_needed -= states_covered # 将已覆盖的州从需要的州列表中删除
    final_stations.add(best_station) # 将此时最优的广播台添加到final_stations中

# 动态规划
# 动态规划先解决子问题，再逐步解决大问题。每种动态规划解决方案都涉及网格。见算法图解P146.
# 给了两个例子：背包问题（装入价值最大的物品集合），寻找最长公共子序列（P154）
# 在问题可分解为彼此独立且离散的子问题时，就可使用动态规划来解决。
# 需要考虑如何将问题分解为子问题

# eg：剪绳子问题
"""
给你一根长度为n的绳子，请把绳子剪成m段（n,m>1），每段绳子的长度记为k[0],k[1]...k[m],问他们可能的最大乘积是多少？
例如：绳子长度为8的时候，把它剪为三段：2、3、3，此时得到的最大乘积是18.
"""
"""
被剪的每一段都是独立的。在剪第一刀的时候，我们有n-1种选择，剪出来的第一段绳子长度可能为1,2...n-1.
f(n)=max( f(i)*f(n-i) ),其中 0<i<n
n=2,f(2)=1;n=3,f(3)=2
"""
# -*- coding:utf-8 -*-
class Solution:
    def cutLine(self, n):
        # 由于必须做一次切割，所以这两种属于特殊情况
        if n==2:
            return 1
        if n==3:
            return 2
        # 注意！绳子长度<3时，不切割的乘积更大，4的时候切割与不切割刚好相等
        products = range(n+1)  # 此数组存储长度从4开始的绳子最大乘积之和（）
        # products[0] = 0
        # products[1] = 1
        # products[2] = 2
        # products[3] = 3   # 可以省略，刚好相等于range产生的
        # 注意，不能直接写products=[0,1,2,3] 这样分配的list只有四个位置，后面无法存储其他元素：index out of range
        for i in range(4,n+1):  # 考虑绳子长度从4到n的情况
            max = 0
            for j in range(1,i/2+1): # j为第一次割下绳子的长度，循环选出最终最大的
                product = products[j]*products[i-j]
                if max < product:
                    max = product
                products[i] = max
        return products[n]

# 15、二进制中1的个数
"""
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
"""
"""
二进制的位运算一共只有六种：与（&），或（|），异或（^），取反（~）
左移 m << n :表示把m左移n位，相当于乘以2^n 
右移 m >> n :表示把m右移n位，相当于整除2^n 

"""

# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        sum = 0
        for i in range(32):
            temp = n>>i & 1
            sum = sum+temp
        return sum
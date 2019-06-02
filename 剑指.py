import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 3、数组中重复的数字
"""
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。
也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 
例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
但不能修改原来的数组。
最佳复杂度：由于所有数字都在0到n-1的范围内，所以可以将数字放到对应的数字的索引上，具体来说：
循环一次，对于值不等于其下标的元素，若此元素的值等于下标等于其值的数字，则说明找到一个重复数字；若不相等，则
交换此元素与下标等于其值的元素的位置。

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
"""
bool duplicate(int numbers[],int length,int *duplication)
{
	if(numbers==nullptr || length<=0)
	{
		return false;
	}
	
	for(int i=0;i<length;++i)
	{
		if(numbers[i]<0 || numbers[i]>length-1)
			return false;
	}
	for(int i=0;i<length;++i)
	{
		while(numbers[i]!=i) //数字不等于其下标
		{
			if(numbers[i]==numbers[numbers[i]])  //与下标等于其值的数字重复
			{
				*duplication=numbers[i];
				return true;
			}
			int temp=numbers[i]; //交换此数字与下标等于其值的数字的位置
			numbers[i] = numbers[temp];
			nums[temp]=temp
		
		}
	}
"""
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
        while i<=rows and j>=0:  # 在达到左下角之前
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
"""
把A2中所有数字插入到A1中（二者均有序），且保持合并后的数组有序(假设从小到大)
"""
# -*- coding:utf-8 -*-
class Solution:
    def merge1(self,A1,A2):
        j = 0
        if not A1:
            return A2
        if not A2:
            return A1
        for i in A2:  # i=0,for i in range(len(A2)):亦可
            while i > A1[j]:
                j += 1
                if j > len(A1)-1: # 若i比A1中剩下的所有元素都大，那么直接把A2剩余部分extend到A1后面即可
                    A1.extend(A2[A2.index(i):]) # 即i值后面的所有值
                    return A1
            A1.insert(j,i)
            j += 1 # A1中的下一个数一定排在此时插入的i位置后面（有序）
        return A1
"""
合并两个有序的数组
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
            s = [0,1]
            # range生成一个list,xrange生成一个生成器（生成一个取出一个，优化内存）
            for i in xrange(2,n+1): # 生成第2项到第n项
                s.append(s[i-1]+s[i-2])
            return s[n]

"""
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
设n级台阶有f(n)种跳法，第一次跳一步，此时跳法数目为f(n-1);第一次跳两步，此时跳法数目为f(n-2)。
故总的跳法数目为f(n-1)+f(n-2)。
n=1,f(1)=1
n=2,f(2)=2
n=3,f(3)=3  就是裴波那契数列，改一改就好
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
            s = [0,1,2]
            # range生成一个list,xrange生成一个生成器（生成一个取出一个，优化内存）
            for i in xrange(3,number+1): # 生成第3项到第n项
                s.append(s[i-1]+s[i-2])
            return s[number]
"""
用8个2*1的小矩形横着或竖着无重叠地覆盖2*8的矩形，总共有多少方法？
这也是一个裴波那契数列问题。
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？ 
思考： 2*1 1 种； 2*2 2种 2*3 3种 2*4 5种 
dp[n]=dp[n−1]+dp[n−2]
代码与跳台阶完全相同！
"""

"""
变态跳台阶：一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
数列：0 1 2 4 8 16 （是前面所有值的累加，恰好是 pow(2,number-1)  if number <= 0: return 0 ）
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
# 暴力检索
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        l=len(rotateArray)
        if l==0:
            return 0
        if rotateArray[0]<rotateArray[l-1]:  # 若产生了旋转，必然>=。说明没有产生旋转，那么最前面的就是最小的元素
            return rotateArray[0]
        pre = -1
        for num in rotateArray:
            if num < pre:
                return num    # 由于两个子序列均递增，若出现一个比它小的，必为最小值
            pre = num
        return rotateArray[0]   # 如果大家都相等，就返回第一个值吧

# 二分法
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        lenArr = len(rotateArray)
        left, right = 0, lenArr-1
        while left <= right:
            if right - left == 1:       # 这一句很关键，解决了最后一次循环，left始终等于mid的问题
                return rotateArray[right]
            mid = (left + right)/2
            if rotateArray[mid] >= rotateArray[left]: # 说明从left到mid处都是有序的
                left = mid
            elif rotateArray[mid] <= rotateArray[right]: # 说明从mid到right处都是有序的
                right = mid
        return rotateArray[right]


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
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        def dfs(index,row,col):
            if row < 0 or col < 0 or row >= len(board) or col >= len(board[0]): # 在边界之外返回False
                return False
            if word[index] == board[row][col]: # 将目前已找到的路径赋值为#，index为目前正在找的字符的索引
                board[row][col] = '#'
                # 若此时index为最后一个，说明所有的字符都找到了，返回True
                if index == len(word)-1 or \
                    dfs(index+1,row+1,col) or \
                    dfs(index+1,row-1,col) or \
                    dfs(index+1,row,col+1) or \
                    dfs(index+1,row,col-1) :
                    return True
                board[row][col] = word[index]  # 这条大路径行不通，将字符恢复为原来的字符，继续下一个方向的探索
            return False
        # 主程序
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]: # 找到起点
                    if dfs(0,i,j): # 0为index，是目前要找的字母，i，j为起点
                        return True
        return False

# 13、机器人的运动范围

"""
地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，
但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。
但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够到达多少个格子？
"""
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        # 设置一个dict全局变量，用来存储每个坐标的到达情况，能到达value设为1，不能到达设为0，防止重复探索
        global dic   # 注意一定要设置这么一个全局变量！否则会报错：未设置全局变量dic
        dic = {}
        return self.count1(0,0,threshold,rows,cols)

    def count1(self,r,c,thresold,rows,cols):
        # 判断坐标是否超过边界(注意有0和最大值两个边界)，超过边界则不能到达
        if r<0 or r>=rows or c<0 or c>=cols:
            return 0
        if dic.get( (r,c) ):  # 若此坐标已被标记能到达，则return0，不执行下一句对四周探索的语句，避免死循环
            return 0          # dict的get函数得到此key对应的value
        # 判断是否符合阈值
        # map函数对字符串中的每一个字符进行操作，然后返回一个list,这种操作可以适应任何位数
        if sum(map(int,str(r)+str(c))) > thresold: # 之前报错，是因为如果之前rc是个负数的话，int('-')报错
            return 0
        else:
            dic[ (r,c)] = 1
        # 下一句的1表示至少此时的起点格子是能够到达的，因为前面已经包含了不能到达的所有情况，并返回了0
        # 后面加的四段代表 从这些点继续探索的点能到达的个数
        return 1+self.count1(r+1,c,thresold,rows,cols)+self.count1(r-1,c,thresold,rows,cols)\
               +self.count1(r,c+1,thresold,rows,cols)+self.count1(r,c-1,thresold,rows,cols)

"""
迷宫求解和状态空间搜索（位置值为1表示是路，为0表示是墙，无法行走）
"""
"""
基于栈的方法（回溯+递归），下面的代码只能找到一条路径就返回了
"""
dirs = [(0,1),(1,0),(0,-1),(-1,0)] # 对于任何一个位置(i,j)，给它加上dirs[i]等就分别得到了该位置东南西北的四个相邻位置
                                   # pos[0]为i，pos[1]为j
# 递归实现的核心函数如下：
def find_path(maze,pos,end): # pos表示搜索的当前位置
    maze[pos[0]][pos[1]] = 2 # 给迷宫maze的位置pos标2表示 “到过了”
    if pos == end: # 已到达出口
        print pos, # 输出这个位置
        return True # 成功结束
    for i in range(4): # 否则按四个方向顺序探查
        nextp = (pos[0]+dirs[i][0],pos[1]+dirs[i][1])
        if maze[nextp[0]][nextp[1]] == 0: # 更新起点，检查迷宫maze的位置pos是否可行
            if find_path(maze,nextp,end): # 从nextp可达出口
                print pos, # 输出这个点
                # 或者将这些pos存储到一个list中，以[::-1]的方式倒序输出，便是路径
                return True # 成功结束
    return False
"""
基于队列的方法：
"""
qu.enqueue(start) # 起点入队
while not qu.is_empty():
    pos = qu.deque() # 队列先进先出，弹出这个位置
    for i in range(4): # 加入此时弹出位置的四个相邻位置，并在加入的过程中判断这四个位置是否为终点
        nextp = (pos[0] + dirs[i][0], pos[1] + dirs[i][1]) # 列举各位置
        if maze[nextp[0]][nextp[1]] == 0:  # 检查迷宫maze的位置pos是否可行
            if nextp == end:
                return True
            maze[nextp[0]][nextp[1]] = 2  # 给迷宫maze的位置pos标2表示 “到过了”
            qu.enqueue(nextp) # 将这个位置加入队列
            # dict1[nextp] = pos  # 将新位置设为key，并将它的上一个位置设为value，方便寻找路径
"""
具体代码见 P169
基于队列的算法，队列里保存的位置及其顺序与路径无关。用一个dict，每次遇到一个新位置的时候，就将新位置设为key，并将它的上一个位置
记为value。到达终点后，取出dict中终点位置的value1，取出value1的value2，直到取到起点位置（写个while语句），这样就构成了一条路径
"""

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
被剪的每一段都是独立的（离散而独立的子问题）。在剪第一刀的时候，我们有n-1种选择，剪出来的第一段绳子长度可能为1,2...n-1.
n=2,f(2)=1;n=3,f(3)=2
设f(n)为长度为n的绳子能得到的最大乘积，则f(n)=max( f(i)*f(n-i) ),其中 0<i<n，n>4
# 注意！绳子长度<=3时，不切割的乘积更大，4的时候切割与不切割刚好相等，因此当i<4或n-i<4，选择不切割
"""
# -*- coding:utf-8 -*-
class Solution:
    def cutLine(self, n):
        # 由于必须做一次切割，所以这两种属于特殊情况
        if n==2:
            return 1
        if n==3:
            return 2
        products = range(n+1)  # 此数组存储长度从4开始的绳子最大乘积之和（）
        # products[0] = 0
        # products[1] = 1
        # products[2] = 2
        # products[3] = 3
        # 注意，不能直接写products=[0,1,2,3] 这样分配的list只有四个位置，后面无法存储其他元素：index out of range
        for i in range(4,n+1):  # 考虑绳子长度从4到n的情况
            max = 0
            for j in range(1,i/2+1): # j为第一次割下绳子的长度，循环选出最终最大的
                max = max(max,products[j]*products[i-j])
                products[i] = max  # 更新长度为i的绳子的最大乘积！这句话可以放在i循环内。
        return products[n]

# 背包问题（动态规划）
"""
背包问题：一个背包能放入重量为weight的物品，有n件物品，重量分别为w0,w1,w2，能否从中选出若干件物品，其重量之和恰好等于weight呢
可以归结为两个n-1件物品的背包问题：
1、一种是减少了重量，物品种类也减一      2、另一种是针对同样重量，但种类减一
"""
def knap_rec(weight,wlist,n): # 函数的三个参数分别是总重量weight，记录各物品重量的表wlist，物品数目n
    if weight == 0: # 终极状态
        return True
    if weight<0 or (weight > 0 and n < 1): # 重量大于0但已经没有物品可用
        return False
    # 装物品的顺序：从最后的物品依次往前装
    if knap_rec(weight - wlist[n-1],wlist,n-1): # 装上物品n，总的重量也减去此物品的重量，物品数目变为n-1
        print ("Item "+str(n)+":",wlist[n-1]) # 加上此物品，就刚好达到总重量
        return True
    if knap_rec(weight,wlist,n-1): # 能用n-1件物品达到总重量
        return True
#  那么这里减去的 wlist[n-1] 应该是谁呢？程序会按顺序判断，比如删去最后一个物品返回False之后，程序一层层返回栈，然后
#  执行第4个if语句，把最后一个物品从列表中删除，试试能不能用去掉最后一个物品之后的n-1件物品实现
#  然后装上倒数第二个物品，同理循环之！妙啊！
weight = 10
wlist = [2,2,6,4,5]
knap_rec(weight,wlist,5)
"""
输出：
Item 3: 6
Item 4: 4
Out[21]: True
"""

# 15、二进制中1的个数
"""
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
"""
"""
二进制的位运算一共只有六种：与（&），或（|），异或（^），取反（~）
左移 m << n :表示把m左移n位，相当于乘以2^n 
右移 m >> n :表示把m右移n位，相当于整除2^n ；与c++不同，python的右移左边填充的是0，不是符号位。

"""

# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        sum = 0
        for i in range(32):
            temp = n>>i & 1 # 1是00000001，因此只验证最后一格是不是1
            sum = sum+temp
        return sum

# 16、数值的整数次方
"""
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
"""
"""编程实现指数的计算，核心语句：for循环，乘上自己exponent次。考虑指数的特殊情况：为0,...指数为负，取个倒数。
base是个浮点数，考虑几个特殊情况。有的时候"""

# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base==0:      # 错误情况，返回False
            return False
        if exponent == 0:
            return 1
        result = 1
        for i in range(abs(exponent)):
            result = result*base
        if exponent<0:
            return 1/result
        return result

# 17、打印从1到最大的n位数
class Solution:
    def PrintToMax(n):
        # write code here
        max = pow(10, n)
        for i in xrange(1, max):
            if i % 10 == 1:    # 一行输出10个数
                print '\n'
            print ('%d' % i),  # 加了个逗号，输出就在同一行了

# 19、正则表达式匹配
"""
请实现一个函数用来匹配包括'.'和'*'的正则表达式。字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）
本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
"""

# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        # 先考虑两种简单情况，pattern长度为0时
        if (len(s) == 0 and len(pattern) == 0):
            return True
        if (len(s) > 0 and len(pattern) == 0):
            return False
        # 考虑 pattern长度大于1且第二个字符为*（意味着*前面的字符可以出现任意次）
        if (len(pattern) > 1 and pattern[1] == '*'):
            # s不为空且第一个字符匹配
            if (len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.')):
                # 三种情况：前面的字符出现了，前面的字符没出现，
                return (self.match(s, pattern[2:]) or self.match(s[1:], pattern[2:]) or self.match(s[1:], pattern))
            else:
                # 第一个字符没匹配，但由于第二个是*，所以可以把pattern的第一个字符当做出现0次，重新开始匹配
                # s长度为0时也是一样，重新开始匹配，若pattern后面全为*则返回True，否则返回False
                return self.match(s, pattern[2:])
        # 只能放在下面...
        if (len(s) > 0 and (pattern[0] == '.' or pattern[0] == s[0])):# 当二者均不为空且第一个字符匹配的时候
            return self.match(s[1:], pattern[1:])
        return False

# 20、表示数值的字符串
"""
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
"""
"""
xswl，用python写，字符串能转为浮点数类型的就是数值，转换不了的就表示不了数值哈哈没毛病
"""
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        try:  # 若出现转换错误，转到except处执行
            p = float(s)
            return True
        except:
            return False

# 21、调整数组顺序使奇数位于偶数前面
"""
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
思路：定义两个指针P1，P2：P1指向数组的最后一个数字，它只向后移动，P2指向数组的最后一个数字，它只向前移动；在P1、P2相遇之前，
P1总是位于P2的前面，若P1指向的数字是偶数，P2指向的数字是奇数，则交换着两个数字。
"""
"""
加上一个条件：保证奇数和奇数，偶数和偶数之间的相对位置不变。常规解法如下：
"""
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        qi=list()
        ou=list()
        for i in list(array):
            if i%2==0:
                ou.append(i)
            else:
                qi.append(i)
        qi.extend(ou)
        return qi


# 29、顺时针打印矩阵

"""
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字(见剑指offer P182)
"""
# -*- coding:utf-8 -*-
class Solution(object):
    def generateMatrix(self, n):
        if not n:
            return []
        res = [[0 for _ in xrange(n)] for _ in xrange(n)]
        left, right, top, down, num = 0, n-1, 0, n-1, 1  # num为要填充到matrix中的第一个数字
        while left <= right and top <= down:
            for i in xrange(left, right+1):
                res[top][i] = num
                num += 1
            top += 1
            # 实际上下面三个for循环的出现都是有条件的
            # if top<=down: 但是不加这个判断，也不影响结果
            for i in xrange(top, down+1):
                res[i][right] = num
                num += 1
            right -= 1
            for i in xrange(right, left-1, -1):
                res[down][i] = num
                num += 1
            down -= 1
            for i in xrange(down, top-1, -1):
                res[i][left] = num
                num += 1
            left += 1
        return res
# 30、包含min函数的栈（举例让抽象问题具体化）
"""
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
"""
"""
建立一个数据栈与辅助栈，每压入一个元素到数据栈中，将数据栈中最小的放在辅助栈中，
每弹出一项，若是最小项，则将辅助栈栈顶元素弹出。
"""

# -*- coding:utf-8 -*-
class Solution:
    # 首先定义两个栈
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, node): # 执行入栈操作
        # write code here
        self.stack.append(node)
        if not self.min_stack or node <= self.min_stack[-1]: # 若加入的元素<=辅助栈栈顶，则入辅助栈
            # 注意：写，if self.min_stack is None 报错了！这种复杂的数据结构，能用not就用not
            self.min_stack.append(node)
    def pop(self):  # 执行出栈操作
        # write code here
        # 若数据栈弹出的那一项碰巧是辅助栈栈顶（最小值），则弹出辅助栈栈顶（相当于更新此时的最小值）
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()   # 这个pop函数前面没有self.，表明调用的是基本库中的pop函数
        self.stack.pop()
    def top(self):
        # write code here
        # 不是弹出，只是显示栈顶
        return self.stack[-1]
    def min(self): # 显示此时的最小值
        # write code here
        return self.min_stack[-1]

# 31、栈的压入、弹出序列
"""
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是弹出序列。
（注意：这两个序列的长度是相等的，压入顺序不变，但可以压入部分元素后，再弹出某元素）
"""
"""
模拟第一个序列的压入，同时检查栈顶是否为弹出序列的第一项：若是，则j=j+1，且执行stack.pop()
"""
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here10
        if not pushV or len(pushV) != len(popV):
            return False
        stack = []
        j = 0
        for i in range(len(pushV)):
            stack.append(pushV[i])
            while stack and stack[-1] == popV[j]:
                j = j + 1
                stack.pop()
        # 等循环结束后再来判断stack是否为空，不能写在循环内部
        if not stack:
            return True


# 38、字符串的排列
"""
输入一个字符串,按字典序打印出该字符串中字符的所有排列（考虑顺序，长度为n）。
例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
"""
"""
思路：把字符串分成两部分：一部分是字符串的第一个字符，另一部分是第一个字符以后的所有字符（阴影区域）。
接下来求阴影区域的字符串的排列。最后，拿第一个字符和它后面的字符逐个交换。
"""
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        if len(ss) <= 0:
            return []
        L = list() # L用来存放最后的字符串
        path = '' # path是已排好的组合
        self.connect(ss,path,L) # ss是剩下的待拾取的字符组成的字符串

        uniq = list(set(L))  # 由于可能会有重复的字符，也就有重复的字符串，需要转为set，再转为list
        return sorted(uniq)    # 并按序排列

    # 每次拿出一个字符，然后将path+此时拿出的字符，剩下的字符，L作为新的参数传入connect函数中
    # 最后一次，ss[i]取到了最后一个字符，此时调用的函数ss变为''，说明递归到了最底层，将path添加到L中
    # 随后返回上一级函数，取出另一个字符；不断返回，直到回到第一级并结束循环
    def connect(self,ss,L,path):
        if ss=='':
            L.append(path)
        else:
            for i in range(len(ss)):
                self.connect(ss[:i]+ss[i+1:],path+ss[i],L)

"""
举一反三！！！
若面试题是按照一定的要求摆放若干个数字，则可以求出这些数字的所有排列，然后一一判断每个排列是不是满足题目给定的要求
例如：P219，正方体顶点放置数字问题，8皇后问题！！！
定义一个数组[8]，数组中第i个数字表示位于第i行的皇后的序号，将数组中的8个数字分别用0-7初始化（必不同列），然后对数组进行全排列
只需判断每一个排列对应的8个皇后是不是在同一条对角线上。
"""

"""
若改为求字符的所有组合（不考虑顺序，长度为1-n），abc的组合有：a,b,c,ab,ac,bc,abc。ab和ba属于同一组合。
"""
""""
https://blog.csdn.net/github_36669230/article/details/58605959
第一种方法：基于位图。 
以输入3个字符a、b、c为例： 3个字符，可以用3个位来表示，从右到左的每一位分别用来代表a、b、c，该位为1表示取该元素，
该位为0表示不取该元素。例如如组合a表示为001，组合b表示为010，组合ac表示为101，组合abc表示为111，ab和ba的表示都是110，
而000是没有意义的，所以总共的结果就是2^n-1种。
因此，我们可以从值1开始循环到2^n-1，输出每个值所代表的组合即可。
"""
# -*- coding:utf-8 -*-
class Solution:
    def Combination(self, ss):
        # write code here
        if len(ss) <= 0:
            return []
        L = list() # 用来存放最后的字符串
        n = len(ss)
        for i in xrange(1,2**n): # 依次输出值1到2^n-1所代表值的组合，1,2,3执行位运算之后变为i=001,010,011等等
            path = ''  # 用来存储本次的组合
            for j in xrange(n):  # 判断第j位是否为1
                temp = i
                # 若i的第j位为1，则字符串第j位对应的字符append到path中
                if temp & 1<<j:  # 或者 temp & 2**j: 即1左移j位，产生：001,010,100等
                    path = path + ss[j]
                # &是位运算，and是逻辑运算。取出temp中为1的位
            # 确定哪些位数为1之后，取出哪些位数的字符，并添加到L中
            L.append(path)
        return L
c = Solution()
print c.Combination('abcd')


"""
第二种方法：采用动态规划来解决。
思路：输入n个字符，则能构成长度为m的组合，1<=m<=n，我们将这n个字符分成两部分：第一个字符和其余的所有字符。
若组合中包含第一个字符，则下一步在剩余的字符里选择m-1个字符，若组合里不包括第一个字符，则下一步在剩余的字符里选m个字符
这样，就分解为了两个子问题：n中求m个字符的组合，n中求m-1个字符的组合
# Python实现根据 https://blog.csdn.net/MDreamlove/article/details/79528836 改编而来，代码运行报错了
"""
# # -*- coding:utf-8 -*-
# class Solution:
#     def Combination(self, ss):
#         # write code here
#         if len(ss) <= 0: # if str==None or len(str) == 0
#             return []
#         global final
#         final = list() # 用来存放最后的字符串
#         L=list()  # 用来存放长度为num个字符时的组合（里面包含多个list）
#         for j in range(1,len(ss)+1):
#             self.connect(ss, 0, j, L) # 从字符串中第一个字符开始，依次取num个（构成长度为num的组合），1 <= num <=len(ss)
#         return final
#
#     def connect(self, str, begin, num, L):
#         # if str==None or len(str) == 0: # 我觉得可以去掉
#         #     return
#         if num==0:  # num为0，说明已经凑够了num个字符，直接输出并返回
#             final.extend(L)
#         if begin > len(str)-1: # 表明上次已经选到了最后一个字符，直接return
#             return
#         # 考虑两种情形
#         L.add(str[begin])   # 选中当前字符
#         self.connect(str,begin+1,num-1,L)
#         L.remove(len(str)-1)   # 不选中此字符
#         # 当前（位置）字符未被选中，删去（上面两行添加了的）,然后查找其他组合
#         self.connect(str, begin + 1, num, L)


# 39、数组中出现次数超过一半的数字（分区思想，寻找中位数）
"""
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

思路1：若把这个数组排序，那么数组的中位数一定就是那个出现次数超过数组长度一半的数字，即长度为n的数组中第n/2大的数字。
先随机选择一个数字，然后调整数组中数字的顺序，使得比它小的都在左边，比它大的都在左边。若此时这个数字的下标刚好是n/2，
那么这个数字就是数组的中位数；若它的下标大于n/2，那么中位数应该位于它的左边；若它的下标小于n/2，那么中位数位于它的右边，
则我们可以接着在它的右边部分的数组中查找。这是一个典型的递归过程。
思路2：数组中有一个数字出现的次数超过数组长度的一半，也就是说它出现的次数比其他数字出现的次数的和还要多。在遍历数组的时候
保存两个值，一个是数组中的一个数字，另一个是次数。当我们遍历到下一个数字的时候，若下一个数字和我们之前保存的数字不同，则
次数减一，若次数为0.我们需保存下一个数字，并将次数设为1；若下一个数字和我们之前保存的数字相同，则次数加1.最终要找的数字
肯定是最后一次保存的数字。最后还要在遍历数组一次，统计这个数字的频次，若检查之后发现并没有达到数组长度的一半，则返回None。
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

# 40、最小的k个数（分区思想，寻找第k大的数）
"""
输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4.（注意k>n的情况,则输出[] ）
先排序，再取 tinput[:k].复杂度为O(nlogn)
实现复杂度为O(n)的思路：Partition，上一题找出中位数，这里要找到数组第k大的数，这个数的左边(包含自己)就是最小的k个数。
"""
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if tinput is None:
            return
        if k > len(tinput):
            return []
        tinput.sort()
        return tinput[:k]

# 41、数据流中的中位数
"""
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
"""
# -*- coding:utf-8 -*-
class Solution:
    # 首先，给这个类加一个初始化的程序，定义一个data,先用insert将num传到data中，然后对data进行各种操作
    def __init__(self):
        self.data = []

    def Insert(self, num): # 数据流读入一个新的数据num
        # write code here
        self.data.append(num)
        self.data.sort()

    def GetMedian(self,data):  # 记得将data作为一个参数传入！！！
        # write code here
        length = len(self.data)
        if length%2 == 0:
            return (self.data[length//2] + self.data[length//2-1])/2.0
            # // 整数除法,返回不大于结果的一个最大的整数./是浮点数除法
            # 6//2=3，表示数组中第4个数字。data[length//2-1]也就是数组中第3个数字
        else:
            return self.data[length//2]
        # 7//2=3，表示数组中第4个数字

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
        max1 = array[0] # 若array全为负数，则返回array中一个最大值，所以这里用max1来存储数组中最大的值
        for i in range(len(array)):
            if array[i] > max1:
                max1 = array[i]
            sum += array[i]
            # 仅当sum>=0时，才保持这个序列，否则重新开始计数。即使加上去sum变小了，也要继续记录，后面会加个更大的数，但不存到max中
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
        # 第一种做法
        l = ''
        for i in range(1, n + 1):
            l += ''.join(str(i))
        return l.count('1')
        # 第二种做法
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
        for i in xrange(n):       # 生成0,1,2,3.。。，10,11,12。。。实际上第n位肯定远远没到第n个数字
            sum += len(str(i))   # 位数之和，如果刚好等于n，则就是i的第一位
            if sum >= n: # 说明i的某一位就是我们要寻找的第n位对应的数字
                # 如果刚好等于n，则就是i的第一位，即i[0],sum为加上第n个数字所有位数的，为i[sum-n].注意，i此时为int，要先转为str
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
        # 定义比较规则：谁排在前面得到的值越大，谁就排在前面
        cmp1 = lambda n1,n2 : int( str(n1)+str(n2) ) - int( str(n2)+str(n1) )
        array = sorted(numbers,cmp=cmp1) # 输出的array为[321, 32, 3]
        return ''.join([str(i) for i in array])

"""
# 注：sorted返回副本（sort在本地进行排序），cmp参数用于比较的函数
sort函数只能用于list，sorted函数可用于任何可迭代的序列
(1)按照元素长度排序
L = [{1:5,3:4},{1:3,6:3},{1:1,2:4,5:6},{1:9}]
def f(x):
    return len(x)
L.sort(key=f)
print L
[{1: 9}, {1: 5, 3: 4}, {1: 3, 6: 3}, {1: 1, 2: 4, 5: 6}]

(2)按照每个字典元素里面key为1的元素的值排序
L = [{1:5,3:4},{1:3,6:3},{1:1,2:4,5:6},{1:9}]
def f2(a,b):
    return a[1]-b[1]
L.sort(cmp=f2)
print L
"""

# 46、把数字翻译为字符串（动态规划）
"""
给定一个数字，按照如下规则翻译成字符串：0翻译成“a”，1翻译成“b”...25翻译成“z”。一个数字有多种翻译可能，
例如12258一共有5种，分别是bccfi，bwfi，bczi，mcfi，mzi。实现一个函数，用来计算一个数字有多少种不同的翻译方法。

两个数字能被翻译为两个字符或一个字符的条件是：两个数字组成的数字在10-25之间
定义函数f(i)表示为从第i位数字开始的不同翻译的数目，那么 f(i)=f(i+1)+g(i,i+1)*f(i+2)。从后面往前走。倒数第二位开始。
当第i位和第i+1位两位数字拼接起来的数字在10-25的范围内时，函数g(i,i+1)的值为1，否则为0。
c++版见此： https://www.jianshu.com/p/80e1841909b7   https://blog.csdn.net/xy_cpp/article/details/79000901
"""
# -*- coding:utf-8 -*-
class Solution:
    def getTranslationCount(self, numbers):
        if numbers < 0:
            return 0
        if numbers == 1:
            return 1
        return self.getCount(numbers)

    def getCount(self,numbers):
        last = 0
        current = 1
        length = len(str(numbers))
        # 注意位数要从后往前，从倒数第二位开始
        for i in range(length-2,-1,-1):
            if int(str(i)+str(i+1)) < 26:
                g = 1
            else:
                g = 0
            temp = current
            current = current + g*last  # 本次=上次+g*上上次，上次赋值为本次
            last = temp       # 上上次 赋值为 上次
        return current


# 47、礼物的最大价值
"""
在一个 m*n 的棋盘中的每一个格都放一个礼物，每个礼物都有一定的价值（价值大于0）.你可以从棋盘的左上角开始拿各种礼物，
并每次向右或者向下移动一格，直到到达棋盘的右下角。给定一个棋盘及上面个的礼物，请计算你最多能拿走多少价值的礼物？
"""
"""
在动态规划求解这个问题的时候，我们找出到达每一行中每个位置累积礼物价值的最大值，只与它上边和左边的值有关，不断更新
和算法图解中的背包问题不同，这里给出了每个格子的物品value，只需要一个与原始矩阵列数相等的一维向量来存储中间计算的值即可
https://blog.csdn.net/dugudaibo/article/details/79678890
背包问题列的间距为物品重量的最小公倍数，用P142的公式计算！！同样需要维护1行4列的矩阵！
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
                if i>0: # 确保存在up
                    up = temp[j]  # 上面框框的值就是此时j列的值，此时尚未更新
                if j>0: # 确保存在left
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
思路：遍历字符串中的每一个元素。
借助一个dict来存储某个元素最后一次出现的下标（实际上存储的是在目前的序列中出现的下标）。
用一个整形变量存储当前无重复字符的子串开始的下标。
将此时最长的序列长度（或最长的序列）与上一轮的长度对比，最长的存储于max1中，并不断更新
https://blog.csdn.net/yurenguowang/article/details/77839381
"""
class Solution:
    def lengthOfLongestSubstring(self, s):
        # write your code here
        if s is None or len(s) == 0:
            return 0
        d = {}
        start = 0
        maxl = 1
        for i in range(len(s)):
# 若从某start开始的子串中出现了重复字符注意是在start之后！可能会有已不在序列中的字符重复了（第三条语句会更新此元素在序列中出现的坐标）
            if s[i] in d and d[s[i]] >= start:
                start = d[s[i]] + 1   # 注意这里 d[s[i]] 存储的位置是已重复的字符，相当于把之前那个被重复的字符移出去
            d[s[i]] = i # 记录下此元素本次出现的坐标
            tmp = i - start + 1 # 此时序列的长度
            maxl = max(maxl,tmp) # 将此时最长的序列长度存储在max1中
        return maxl

# 49、丑数
"""
把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。
 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
"""
"""
丑数：如果一个数%2==0，就连续除以2；%3==0，就连续除以3；%5==0，就连续除以5。若最后得到的是1，那么这个数就是丑数。
逐一遍历时间消耗太大。试着找到一种只计算丑数的方法。丑数应该是另一个丑数乘以2、3或5的结果。
因此可以创建一个数组，里面的数字是排序好的丑数。观察 1 1*2 1*3 2*2 1*5 2*3  
比较min(uglylist[index2]*2 ,uglylist[index3]*3 ,uglylist[index5]*5)
"""
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if (index <= 0):
            return 0
        uglylist = [1]
        # index2:肯定存在某个丑数，排在它之前的每个丑数乘以2得到的结果都会小于已有的最大丑数，在它之后的每个丑数乘2的结果又太大
        # 说人话：index2就是丑数列表中由某一个丑数乘以2得到的最新的索引
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
# 可以利用一个OrderedDict（key为位置，value为该位置的字符），若遍历到某字符 in d，则value+1，否则，增加一个新key，并将其value设为1
# 在d中第二次遍历，扫到的第一个value为1的
"""

import collections
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        d = collections.OrderedDict()
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
# 两个OrderedDict，如果其里面item顺序不同那么Python也会把他们当做是两个不同的对象
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
    def Insert(self, char):  # 字符流计数程序，读入一个char，则添加到s中，并对这个字符的数目+1，若不存在则添加这个key
        # write code here
        self.s = self.s + char
        if char in self.dict1:
            self.dict1[char] = self.dict1[char] + 1
        else:
            self.dict1[char] = 1
    def FirstAppearingOnce(self):
        # write code here
        for i in self.s:
            if self.dict1[i]==1:
                return i
        return '#'

# 51、数组中的逆序对  逐一比较某一个数和它后面的所有数的复杂度为O(n*n)，优化方法见 P268（没看！！！）
"""
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
题目保证输入的数组中没有的相同的数字
数据范围：
	对于%50的数据,size<=10^4
	对于%75的数据,size<=10^5
	对于%100的数据,size<=2*10^5
例子：2 1 3 4的逆序对为1，逆序对数=调整为有序要进行的交换次数
"""
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        copy = []
        count = 0
        for i in data: # 相当于实现了一个对data的深复制
            copy.append(i)
        copy.sort()

        for i in range(len(copy)): # 记录下有序序列中索引与原始序列索引的差值，注意每次记录一个差值之后要remove此数
            # 由于每次都要remove此数，所以实际上每次比较的时候，copy[i]在copy中的索引都是0
            count += data.index(copy[i]) - 0
            data.remove(copy[i])

        return count % 10000000007

# 第二种方法，用空间换时间，用归并排序来做
# 把原数据分成两个数组，每次取两个数组中的最小值放入一个新的数组中，直到其中一个数组全部取完
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        length = len(data)
        if data == None or length <= 0:
            return 0
        copy = [0] * length
        for i in range(length):
            copy[i] = data[i]

        count = self.InversePairsCore(data, copy, 0, length - 1)
        return count % 10000000007

    def InversePairsCore(self, data, copy, start, end):
        if start == end:
            copy[start] = data[start]
            return 0
        length = (end - start) // 2
        left = self.InversePairsCore(copy, data, start, start + length)
        right = self.InversePairsCore(copy, data, start + length + 1, end)

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


# 53、在排序数组中查找数字
"""
统计一个数字在排序数组中出现的次数。
"""

# 思路：既然是排序数组，直接用二分法啊    start = 0  end = len(data)-1
# 当找到这个数字k之后，查看上一个数字，若不是k，说明这是第一个k，若是k则进入递归，直到找到第一个k的位置，first=其索引
# 同理可寻最后一个k的位置，end=其索引，次数=end-first+1
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        return data.count(k)

class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        if len(data)>0:
            first = self.get_first(data,k,0,len(data)-1)
            last = self.get_last(data,k,0,len(data)-1)
            if first>-1 and last>-1:
                number = last - first + 1
        return number

    def get_first(self,data,k,start,end):
        if start>end: # 二分法进行的条件 while start<=high:
            return -1
        mid = (start+end)/2
        findk = data[mid]
        if findk == k:
            if mid==0 or mid>0 and data[mid-1]!=k: # 是第一个k
                return mid
            else:
                end = mid-1 # 向前探索第一个k的位置
        elif findk>k:
            end = mid-1
        else:
            start = mid+1
        return self.get_first(data,k,start,end)

    def get_last(self,data,k,start,end):
        if start>end:
            return -1
        mid = (start+end)/2
        findk = data[mid]
        if findk == k:
            if mid==len(data)-1 or mid<len(data)-1 and data[mid+1]!=k:
                return mid
            else:
                start = mid+1
        elif findk>k:
            end = mid-1
        else:
            start = mid+1
        return self.get_last(data,k,start,end)

# 循环写法（节省空间）
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        if not data:
            return 0
        if len(data) == 1 and data[0] != k:
            return 0
        left = 0
        right = len(data) - 1
        first_k = 0
        while left <= right:
            mid = (left + right) // 2   # //为整除
            if data[mid] < k:
                left = mid + 1
            elif data[mid] > k:
                right = mid - 1
            else:
                if mid == 0:
                    first_k = 0
                    break;
                elif data[mid-1] != k:
                    first_k = mid
                    break;
                else:
                    right = mid - 1
        left = 0
        right = len(data) - 1
        last_k = -1
        while left <= right:
            mid = (left + right) // 2
            if data[mid] < k:
                left = mid + 1
            elif data[mid] > k:
                right = mid - 1
            else:
                if mid == len(data) - 1:
                    last_k = len(data) - 1
                    break;
                elif data[mid+1] != k:
                    last_k = mid
                    break;
                else:
                    left = mid + 1
        return last_k - first_k + 1

"""
在范围0~n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字

直观思路：求解这n个数字的和-这n-1个数字的和，它们的差就是缺少的那个数字
利用递增：假设m不在数组中，那么m+1就处在下标为m的位置，可以发现：m正是数组中第一个值和下标不相等的元素的位置
找到某一个值和下标不相等的元素后，向前继续探索，直到找到第一个二者不相等的元素 def get_first(self,data,k,start,end):
"""
# 二分法实现:
while start<=end:
    mid = (start+end)/2
    if data[mid]==mid:
        start = mid+1  # 在右边寻找
    elif mid==0 or data[mid-1]==mid-1:
        return mid  # 找到了
    else:
        end = mid -1 #

"""
from __future__ import division 存在，此时/代表浮点数除法，//代表整数除法，即与Python3.x的版本相同
若此句不存在，那么/与//均代表整数除法，无区别
"""
"""
单调递增数组里的每个元素都是整数且唯一，找出数组中任意一个数值等于其下标的元素
那么:前面的必定<=其下标，后面的>=其下标
"""
#二分法：
if data[mid]==i :
    return data[mid]
if data[i]>i :
    end = mid-1
else :
    start=mid+1


# 56、数组中数字出现的个数
"""
题目1：在一个数组中除了一个数字只出现一次之外，其他数字都出现了2次，请找出那个只出现了一次的数字。
要求：线性时间复杂度O(N)，空间复杂度为O(1)
思路：用位运算来解决XOR异或来解决该问题。由于两个相同的数字的异或结果是0，我们可以把数组中的所有数字进行异或操作，
结果就是唯一出现的那个数字。
比如：12233,0001 0010 0010 0011 0011 这五个数字做一个异或，可以改变 异或 的顺序，相同的全变为0，所以最终只留下0001，即1
同理适用于：
数组中只出现一次的数字：一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个只出现一次的数字。
java:
public int singleNumber(int[] nums) {  
        int ans =0;  
      
        int len = nums.length;  
        for(int i=0;i!=len;i++)   # 第一个数和第二个数做异或（位运算），得到的结果再与第三个数做异或
            ans ^= nums[i];  
        return ans;    
    }  

题目2：数组中唯一出现一次的两个数字（medium）
在一个数组中除了2个数字只出现一次之外，其他数字都出现了2次，请找出两个只出现了一次的数字。

思路：从头到尾异或数组中的每个数字，可以得到只出现1次的两个数字的异或结果。
从异或结果中，找到右边开始第一个不为0的位数（利用右移位的指令+与操作），记为n。
我们将数组中所有的数字，按照第n位是否为0，分为两个数组。
每个子数组中都包含一个只出现一次的数字，其它数字都是两两出现在各个子数组中。那么结合题目一，我们已经得出了答案。
 
 
题目3：在一个数组中除了一个数字只出现一次之外，其他数字都出现了3次，请找出那个只出现了一次的数字。

三个相同的数字异或之后还是本身。转换思路：如果一个数字出现三次，那么他的二进制表示的每一位(0或者1)也出现3次。
如果把所有出现3次的数字的二进制表示的每一位都分别相加起来，那么每一位的和都能被3整除！！！
我们把数组中所有数字的二进制表示的每一位都加起来。如果某一位的和能被3整除，那么那个只出现一次的数字二进制表示中对应的那一位为0，
否则就是1。从而可以得到只出现一次的那个二进制数的各个位数的0和1情况，转为整数，就是我们要找的数。

"""

# 题目2：
# 可以建立一个新的list，第一次遇到该字符则append进去，第二次遇到则remove，由于出现了偶数次，最后一定被移出去
# 也可以直接collections.Counter(array)计数，这两种复杂度都比较高。直接忽略吧
# -*- coding:utf-8 -*-
# class Solution:
#     # 返回[a,b] 其中ab是出现一次的两个数字
#     def FindNumsAppearOnce(self, array):
#         # write code here
#         tmp = list()
#         for i in array:
#             if i in tmp:
#                 tmp.remove(i)
#             else:
#                 tmp.append(i)
#         return tmp

# 题目3：
# # -*- coding:utf-8 -*-
# import collections
# class Solution:
#     def FindNumsAppearOnce(self, array):
#         # write code here
#         tmp = collections.Counter(array)  # 返回一个dict
#         for k, v in tmp.items():
#             if v == 1:
#                 return k


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
        i = 0  # start
        j = len(array) - 1  # end
        while i < j:
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
等于的时候下一步移动end或start均可。(探索未知)循环进行的条件：while small < (tsum+1)/2，否则光是start+end就大于tsum了
"""


# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        if tsum < 3:
            return []
        start = 1
        end = 2
        sum = 3  # 序列所有值累加到这，没必要写一个子函数，for循环
        output = list()
        while start < (tsum + 1) // 2:
            if sum == tsum:
                output.append(range(start, end + 1))
                end += 1
                sum += end
            elif sum < tsum:
                end += 1
                sum += end  # 更新sum
            else:
                sum -= start
                start += 1  # 起始位置+1，但注意这一项已经加在sum里面了
        return output


# 58、翻转字符串
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
        return ' '.join(l[::-1])  # 切片：起点坐标，终点坐标向正或负方向多走一步，方向

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


# 59、队列的最大值
"""
给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}

# 思路:滑动窗口可以看成一个队列，当窗口滑动时，处于窗口的第一个数字被删除，同时在窗口末尾添加一个新的数字
# 之前题30用O(1)时间得到最小值的栈，题9讨论了如何用两个栈实现一个队列。这样将时间复杂度降低到了O(n)
import collections
d = collections.deque()
d.appendleft(1)  # 入队
print d.pop()    # 出队
下面直接在list上操作，取得每次窗口滑动时的第一个索引并往后数size个，取这一范围的最大值
"""
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if size <= 0:
            return []
        res = []
        for i in xrange(0, len(num) - size + 1):  # 每次窗口滑动时候停留的第一个索引值
            res.append(max(num[i:i + size])) # 取滑动窗口的最大值
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
    PrintProbability(6);  % string s;cin>>s;count<<res.size()
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
        m = c[0]  # 统计0的个数（大小王）
        new_list = [i for i in numbers if i > 0]
        new_list.sort()
        n = 0
        for j in range(len(new_list) - 1):
            if new_list[j + 1] - new_list[j] > 0:
                n += new_list[j + 1] - new_list[j]  # 统计空缺值
            else:
                return False  # 若有相等元素，返回False
        if n <= m + len(new_list) - 1:
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
        while len(res) > 1:
            i = (i + m - 1) % len(res)  # 每次从起点i出发向前走m-1步
            res.pop(i)
        return res[0]

# 63、股票的最大利润
"""
一只股票在某些时间节点的价格为{9,11,8,5,7,12,16,14}。若在价格为5的时候买入并在价格为16的时候卖出，则能收获最大的利润11
寻找数组中拥有最大差值的一对数
"""
"""
定义函数 diff(i)为当卖出价为数组中第i个数字时可能获得的最大利润。显然，在卖出价固定时，只要记住
前面i-1个数字时的最小值即可。
"""

# -*- coding:utf-8 -*-
class Solution:
    def maxbenefit(self, time):
        if not time or len(time) == 1:
            return 0
        buy = time[0]
        max_benifit = time[1] - time[0]
        for i in range(2, len(time)):
            buy = min(buy, time[i - 1])
            benefit = time[i] - buy
            max_benifit = max(max_benifit, benefit)
        return max_benifit

c = Solution()
print c.maxbenefit([9, 11, 8, 5, 7, 12, 16, 14])

# 64、求1+2+3+...+n
"""
求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
"""


# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        def f(x, y):
            return x + y
        return reduce(f, range(n + 1))
    # map是对序列中的每个元素执行前面的函数操作
    # reduce把一个函数（必须接受两个参数）作用在序列上，reduce把结果继续和序列的下一个元素做累积计算
    # like this : reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)


# 65、不用加减乘除做加法
"""
写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
利用list的sum函数
思路：只能利用位运算了。
比如：5+17。三步走：第一步只做各位相加不进位，这一步得到12；第二步做进位，进位的值是10；第三步相加，22
尝试把二进制的加法用位运算来替代。第一步，就是异或运算；第二步，仅当1+1时会产生进位，可以当作是两个数先做位与运算，然后
再向左移动一位；第三步的相加的过程，依然是重复前面两步，直到不产生进位为止。
"""

# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        # write code here
        s = []
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
        B = [None] * num  # 否则后面B[i-1]会报错
        B[0] = 1
        # B[1] = A[0]
        # B[2] = A[0]*A[1]
        # B[n] = ...*A[n-1]
        for i in range(1, num):
            B[i] = B[i - 1] * A[i - 1]
        # 计算后面一部分
        # 自下而上,最后才对B[1]进行乘积，而且它是下半部分乘的最多的
        # 保留上次的计算结果乘本轮新的数,因为只是后半部分进行累加
        # 所以设置一个tmp,能够保留上次结果。实际上前面的B[i-1]就承担了tmp的作用
        tmp = 1
        for i in range(num - 2, -1, -1):  # 注意要从倒数第二项开始，因为B[n-1]项已经乘完了，一直乘到第二项A[1]
            tmp *= A[i + 1]
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
        except:  # except Exception as e: 此语句可以捕获与程序退出 sys.exit()相关之外的所有异常
            return 0


# 快速排序
def quick_sort(lst):
    if len(lst) < 2: # 基线条件
        return lst
    else:              # 递归条件
        pivot = lst[0] # 作为基准（pivot）
        # print type(pivot),'1',type([pivot]),'2'  # 前为 int，后为 list
        # 分区（partition）操作
        less = [i for i in lst[1:] if i.key <= pivot.key] # 由所有小于基准值的元素组成的子数组
        greater = [i for i in lst[1:] if i.key > pivot.key]
        # 递归（recursive）排序
        return quicksort(less) + [pivot] + quicksort(greater) # 列表的排序
# 利用栈实现快排的非递归形式
"""
1）栈里边保存什么？
2）迭代结束的条件是什么？
栈里边保存的当然是需要迭代的函数参数，结束条件也是跟需要迭代的参数有关。
对于快速排序来说，迭代的参数是数组的上边界low和下边界high，迭代结束的条件是low == high。
"""
def quick_sort(array, l, r): # 这里按照从大到小的顺序
    if l >= r:
        return
    stack = []
    stack.append(l)
    stack.append(r)
    while stack:
        low = stack.pop(0) # 第一个元素（并从栈中删去）
        high = stack.pop(0) # 第二个元素
        if high - low <= 0:
            continue
        x = array[high]
        i = low - 1
        for j in range(low, high):
            if array[j] <= x:
                i += 1
                array[i], array[j] = array[j], array[i]
        array[i + 1], array[high] = array[high], array[i + 1]
        stack.extend([low, i, i + 2, high])


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
循环一次，对于值不等于其下标的元素，若此元素的值等于 下标=其值 的下标位置上的数字，则说明找到一个重复数字；若不相等，
则将此元素放到 下标=其值 的位置上去，即交换此元素与下标等于其值的元素的位置。

"""
# coding:utf-8
import collections # 需要用到里面的Counter计数器
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]，duplication用来存放重复的数字
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
见leetcode88. Merge Sorted Array 合并两个有序列表（要求合并到某一列表中：从最后开始填充）
Input:
nums1 = [1,2,3,0,0,0], m = 3 ,num1中有足够的空间，length >= m+n
nums2 = [2,5,6],       n = 3
Output: [1,2,2,3,5,6]
最终组成的nums1长度必然为m+n，可以从最后开始比较，最大的放到最后一个位置，依次向前，这样便不会占用掉还未比较的数
"""
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        n1 = m-1
        n2 = n-1
        while n1>=0 and n2>=0:  # 从最后依次向前比较
            if nums2[n2] >= nums1[n1]:
                nums1[n1+n2+1] = nums2[n2]
                n2 -= 1
            else:
                nums1[n1+n2+1] = nums1[n1]
                n1 -= 1
        if n1 < 0:
            nums1[:n2+1] = nums2[:n2+1]
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

# 31、栈的压入、弹出序列
"""
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是弹出序列。
（注意：这两个序列的长度是相等的，压入顺序不变，但可以压入部分元素后，再弹出某元素）

思路：模拟第一个序列的压入，同时检查栈顶是否为弹出序列的第一项：若是，则j=j+1（j指向出栈序列的第一个值），且执行stack.pop()
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
        # 判断stack是否为空
        if not stack:
            return True


# 30、包含min函数的栈（举例让抽象问题具体化）
"""
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
建立一个数据栈与辅助栈，每压入一个元素到数据栈中，若加入的元素<=辅助栈栈顶，则入辅助栈
数据栈每弹出一项，若是辅助栈栈顶元素，则将辅助栈栈顶元素弹出。
"""


# -*- coding:utf-8 -*-
class Solution:
    # 首先定义两个栈
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, node):  # 执行入栈操作
        # write code here
        self.stack.append(node)
        if not self.min_stack or node <= self.min_stack[-1]:  # 若加入的元素<=辅助栈栈顶，则入辅助栈
            self.min_stack.append(node)

    def pop(self):  # 执行出栈操作
        # write code here
        # 若数据栈弹出的那一项碰巧是辅助栈栈顶（最小值），则弹出辅助栈栈顶（相当于更新此时的最小值）
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()  # 这个pop函数前面没有self.，表明调用的是基本库中的pop函数
        self.stack.pop()

    def top(self):
        # write code here
        # 不是弹出，只是显示栈顶
        return self.stack[-1]

    def min(self):  # 显示此时的最小值
        # write code here
        return self.min_stack[-1]


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

    def Insert(self, num):  # 数据流读入一个新的数据num
        # write code here
        self.data.append(num)
        self.data.sort()

    def GetMedian(self, data):  # 记得将data作为一个参数传入！！！
        # write code here
        length = len(self.data)
        if length % 2 == 0:
            return (self.data[length // 2] + self.data[length // 2 - 1]) / 2.0
            # // 整数除法,返回不大于结果的一个最大的整数./是浮点数除法，如果/前后都是int的话，那结果是一样的
            # 6//2=3，表示数组中第4个数字。data[length//2-1]也就是数组中第3个数字
        else:
            return self.data[length // 2]
        # 7//2=3，表示数组中第4个数字

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
故总的跳法数目为f(n-1)+f(n-2)。（简单的动态规划）
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
寻找最小值可以采用二分法，复杂度为O（log n）。二分法找到非排序的那一段，最后一次循环时，以 3 4 5 1 2或3 4 1 2为例，始终都是
5 1或4 1，即当 right-left==1：return rotateArray[right]
"""
# -*- coding:utf-8 -*-
# 二分法
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here)
        left, right = 0, len(rotateArray)-1
        while left <= right:
            if right - left == 1:       # 这一句很关键，解决了最后一次循环，left始终等于mid的问题
                return rotateArray[right]
            mid = (left + right)/2
            if rotateArray[mid] >= rotateArray[left]: # 说明从left到mid处都是有序的
                left = mid
            elif rotateArray[mid] <= rotateArray[right]: # 说明从mid到right处都是有序的
                right = mid
        return rotateArray[right]

# 在包含重复数字的旋转数组中找出某个数的坐标，没有则返回-1
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        N = len(nums)
        l, r = 0, N - 1
        while l <= r:
            while l < r and nums[l] == nums[r]:
                l += 1  # 使左指针和右指针指向不同的数字 # 若没有重复数字，这两句去掉即可
            mid = (l + r) / 2
            if nums[mid] == target:
                return mid
            if nums[mid] >= nums[l]:  # 说明从l到mid这一段是有序的
                if target < nums[mid]:  # 若目标在mid前面 # nums[l] <= target < nums[mid]
                    r = mid - 1   # 实际上此时后面这几个判断有序的语句就失效了，因为一定有序
                else:  # 目标在mid后面
                    l = mid + 1
            elif nums[mid] <= nums[r]:  # 说明mid到r这一段是有序的
                if nums[mid] < target:  # 目标在mid后面 # nums[mid] < target <= nums[r]
                    l = mid + 1
                else:
                    r = mid - 1
        return -1

# 二分法代码
def search(self,array,num):
    left, right = 0, len(array) - 1
    while left<=right:
        mid = (left+right)/2
        if num == array[mid]:
            return mid
        elif array[mid]>num:
            right = mid-1
        else:
            left = mid+1
    return None

35. Search Insert Position  插入某个数字，找到此数字应该放在此有序数组中的位置

"""
用二分法来寻找插入位置，若找到某一个位置的值==此数字，则插入到此位置；若没有找到这个位置，最终得到的start==end，
start就是此时要插入的位置
"""
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        start = 0
        end = len(nums)-1
        while start <= end:
            mid = (start+end)/2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid-1
            else:
                start = mid + 1
        return start

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
            if row < 0 or col < 0 or row >= len(board) or col >= len(board[0]): # (新的)起点在边界之外返回False
                return False
            if word[index] == board[row][col]: # 注意四个方向的搜索是在 if word[index] == board[row][col]:的条件下的！
                board[row][col] = '#'          # 先将目前找到的字符标记为 #
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
                    if dfs(0,i,j): # 0为是目前要找的字母在数组中的下标，i，j为起点
                        return True
        return False

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
test = [[0 for _ in range(n)] for _ in range(m)]
def dfs(i,j,end_x,end_y):
    if i==end_x and j==end_y: # 有一条路径到达了终点，count+1
        C.count += 1
        return # 立即返回，终点始终不会标记为1！
    test[i][j]=1 # 走过的格子标记为1
    for r,c in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
        if r>=0 and r< n and c>=0 and c< m and test[r][c]==0 and state[r][c]>state[i][j]:
            # 由于每次的海拔要比之前高，所以和13题先判断是否要return不同，这里写的是进入dfs的条件
            dfs(r,c)
    test[r][c]=0
dfs(start_x,start_y,end_x,end_y)
print C.count

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
        board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.count = 0  # 定义一个类内变量，实现全局变量的效果
        def dfs(r,c):
            if not (0<=r<rows and 0<=c<cols):
                return
            if board[r][c]!=0:
                return
            if board[r][c]==-1 or sum(map(int,str(r)+str(c))) > threshold:
                board[r][c]=-1
                return
            board[r][c] = 1
            self.count+=1
            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)
        traverse(0,0)
        return self.count

39. Combination Sum  在一个候选列表中找到某些数字的和等于目标值（允许数字重复）
"""
Input: candidates = [2,3,5], target = 8,
A solution set is:[2,2,2,2],[2,3,3],[3,5] 
"""
# dfs，寻找多种可能性的回溯算法哟！
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.res = [] # 相当于定义一个全局变量，可以在类中所有函数中使用
        candidates = sorted(candidates) # 从小的开始选
        self.dfs(candidates,[],target,0) # 候选集，此时数字组合，此时目标值，上一次被放入组合的数字
        return self.res

    def dfs(self,candidates,sub,target,last):
        # candidates始终不变，因为允许重复选择同一个元素
        # sub:已选中的数字，target：8减去已选中的数字，last:上轮选中的数字（本轮选的数字要>=last，否则会导致重复）

        if target == 0:  # 目标变为0，此组可行，加入res
            self.res.append(sub[:])  # 这个地方一定要带切片啊，因为如果不带分片，就是一个浅拷贝（会传入初始空值）
        if target < candidates[0]: # 目标不为0且小于此时最小值，直接return
            return
        # 开始dfs组合
        for n in candidates:
            if n > target: # 由于是排序数组，后面的数字肯定也不行
                return
            if n < last:  # 下次选的要>=last，使得符合的数组前面小后面大，不会重复
                continue
            sub.append(n)
            self.dfs(candidates,sub,target-n,n)
            sub.pop()  # 加入上面返回的元素失败了或成功了，弹出这一项，继续探索，进行下一个循环

216. Combination Sum III
"""
用1-9中的数字，且不能重复使用，组成k个数使得等于n，构成的组合也不能重复
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
"""
class Solution:
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        self.res = []
        nums = range(1, 10)
        self.dfs(nums, k, n, 0, [])
        return res

    def dfs(self, nums, k, n, index, path):
        # edge case
        if k < 0 or n < 0:
            return
        # when reaching the end
        if k == 0 and n == 0:
            self.res.append(path[:])  # 这里写path好像也行
        for i in range(index, len(nums)):
            self.dfs(nums, k - 1, n - nums[i], i + 1, path + [nums[i]])

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

"""
招银网络科技18.2：
考虑你从家出发步行去往一处目的地，该目的地恰好离你val（整数）。
你只能朝向该目的地或者背向该目的地行走，而你行走的必须为单位步长的整数倍，且要求你第N次行走必须走N步。
请就给出目的地离你距离，判断你是否可以在有限步内到达该目的地。如果可以到达的话，请计算到达目的地的最短总步数(不能到达则输出-1)。
距离目的地2， 需要3步：朝向走1，背向走2，朝向走3
"""
# if x == 0:
#     distanceSet.add(x)
#     if val in distanceSet:  # 若val存在于distanceSet中，说明可以到达终点
#         return x
#     else:
#         return find(x + 1, val, distanceSet)  # 主要程序开始：x=1，val，distanceSet为（0，）起点位置为0
"""
宽度优先遍历（利用队列），这样得到的第一个能够到达终点的，就是最优路径，即最短总步数
距离目的地2， 需要3步：朝向走1，背向走2，朝向走3
"""
# -*- coding: utf-8 -*-
val = input()
distanceSet = set()
print(find(0, val, distanceSet))
def find(x, val, distanceSet): # x表示本次走的步数，distanceSet队列存储着目前可以抵达的位置，val为终点的位置
    # 分两种情况考虑：x==0的时候将0作为队列初始值添加到队列中，其实也可以直接将队列定为初始值0，只不过可以考虑一种
    # 起点即终点的特殊情况，直接返回x。
    if x == 0:
        if val==0:  # 若val存在于distanceSet中，说明可以到达终点
            return x
        else:
            distanceSet.add(x)
            return find(x + 1, val, distanceSet)  # 主要程序开始：x=1，val，distanceSet为（0，）起点位置为0
    #         return -1
    elif x > 0:
        dSet = set() # dset用来存储本次能到达的2n个地方
        for d in distanceSet: # len(dSet)=len(distanceSet)*2，在上一次的每个位置上，向前走或向后走x步，更新set
            dSet.add(d + x)
            dSet.add(d - x)
        if val in dSet: # 在上次的位置中，向前或向后走x能到达终点
            return x  # x表示本次走的步数（也就是最终到终点时的步数，宽度优先遍历得到的同时也是最优的步数）
        else: # 注意传入的是x+1，dSet赋给distanceSet，相当于将distanceSet中原来的位置舍弃了，只留下+-x之后的位置
            return find(x + 1, val, dSet)

# 14、贪婪算法

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
直接将相应的参数传入函数就可以了
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

# leetcode322 零钱兑换 （动态规划）
"""
给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。
如果没有任何一种硬币组合能组成总金额，返回 -1。0
Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1

dp[i] = min(dp[i], dp[i - coin] + 1)
 # 金额为i的最小硬币个数 = min( 某一次组合得到的dp[i] , 加上本次的硬币组合得到的 ）
"""
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int] list中每一个数都可以无限次使用
        :type amount: int
        :rtype: int
        """
        dp = [0] + [float('inf')] * amount
        for coin in coins: # 依次取出第一枚硬币，第二枚...
            for i in range(coin, amount + 1): # 从此时最小的硬币金额开始，可以看到，后一项 dp[i - coin] 始终是从dp[0]开始的
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1

# 46、把数字翻译为字符串（动态规划） f(i)=f(i+1)+g(i,i+1)*f(i+2)
"""
给定一个数字，按照如下规则翻译成字符串：0翻译成“a”，1翻译成“b”...25翻译成“z”。一个数字有多种翻译可能，
例如12258一共有5种，分别是bccfi，bwfi，bczi，mcfi，mzi。实现一个函数，用来计算一个数字有多少种不同的翻译方法。

两个数字能被翻译为两个字符或一个字符的条件是：两个数字组成的数字在10-25之间
定义函数f(i)表示为从第i位数字开始的不同翻译的数目，那么 f(i)=f(i+1)+g(i,i+1)*f(i+2)。从后面往前走。倒数第二位开始。
当第i位和第i+1位两位数字拼接起来的数字在10-25的范围内时，函数g(i,i+1)的值为1，否则为0。
c++版见此： https://www.jianshu.com/p/80e1841909b7   https://blog.csdn.net/xy_cpp/article/details/79000901
        # f = [0] * length
        # f[length-1] = 1
        # for i in range(length-2,-1,-1):
        #     if i == length-2:
        #         f[i+1] = 1 + g*f[i+1]
        #         continue
        #     f[i] = f[i+1] + g*f[i+2]
  下面的实现不需要构造一个向量，可以实现O(1)的空间复杂度:（每一项单独处理）+（是否能把第i和第i+1合并处理）这两种情况。
   current = current + g*last  # 本次=上次+g*上上次，上次赋值为本次
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
        last = 1
        current = 1
        length = len(str(numbers))
        # 注意位数要从后往前，从倒数第二位开始
        for i in range(length-2,-1,-1):
            if int(str(numbers)[i]+str(numbers)[i+1]) < 26:
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
        # if rows > cols:
        #     temp = [0]*cols
        #     dp(rows,cols)    # 将下面写的双循环打包为dp函数，下面实际上仅仅是rows > cols的情况
        # else:
        #     temp = [0]*rows
        #     dp(cols,rows)
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

if word_a[i]==word_b[j]:
    cell[i][j] = cell[i-1][j-1]+1
else:
    cell[i][j] = 0  # 最长公共子串P152
    cell[i][j] = max(cell[i-1][j],cell[i][j-1]) # 最长公共子序列P155；将之前公共的次数传递下来。注意先对边缘元素做处理。

64. Minimum Path Sum 最短路径
"""
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.只能向下或右移动。
同样用动态规划来做，只不过此时dp中存储的是此位置上的最小权重路径。dp[i][j] = grid[i][j]+min(dp[i-1][j]+dp[i][j-1])
"""

class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = len(grid[0])
        dp = [ [0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                temp1 = 1e6  # 给temp值赋值一个较大值，当dp[i][j]为边缘值时，会选择较小的那个
                temp2 = 1e6  # 除了dp[0][0]之外，temp1和temp2总有一个不为1e6；对dp[0][0]赋初值grid[0][0]
                if i==0 and j==0:
                    dp[i][j] = grid[0][0]
                    continue
                if i>0: # 确定不是边缘
                    temp1 = dp[i-1][j]
                if j>0:
                    temp2 = dp[i][j-1]
                dp[i][j] = grid[i][j] + min(temp1,temp2)
        return dp[m-1][n-1]


62. Unique Paths 路径个数
"""
m行n列的矩阵，从左上角走到右下角，每次只能向右或向下一共有几种走法
典型的动态规划问题，
"""

class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if m<1 or n <1:
            return 0
        if m==1 or n==1:
            return 1
        # 创建一个m行n列的矩阵列表dp，dp中每一个位置代表达到此位置的路径数（实际上构造一行向量即可）
        dp = [ [0 for _ in range(n)] for _ in range(m)]
        # 初始化边缘
        for i in range(n):
            dp[0][i] = 1   # 到达第一行的各个位置路径数均为1
        for i in range(m):
            dp[i][0] = 1
        for row in range(1,m): # 都是从1开始的
            for col in range(1,n):
                dp[row][col] = dp[row-1][col] + dp[row][col-1]
        return dp[m-1][n-1]

63. Unique Paths II
"""
矩阵中0代表可以通行，1代表不能通行
Input:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
Output: 2
Explanation:
There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
在前一题的基础上需要判断某个位置是不是有障碍物，如果有障碍物，那么到达这个地方的方法是0 
"""

class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0 for col in range(n)] for row in range(m)]
        # 由于矩阵每一处都有可能是障碍，所以将边缘初始化为1行不通了
        # 直接做循环（若某点是障碍，将dp[i][j]设为0，由于没有做边缘的初始化，因此要判断是否为边缘）
        if obstacleGrid[0][0] == 1:
            return 0
        dp[0][0] = 1
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0   # 并直接进入下一次循环
                else:
                    if i > 0: # 判断是否为边界
                        dp[i][j] += dp[i-1][j]
                    if j > 0:
                        dp[i][j] += dp[i][j-1]
        return dp[m-1][n-1]

120. Triangle
"""
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
动态规划：左边的元素只能从上面上一行同一列下来，中间的可以从上一行前一列或同列，右边的只能从前一列
统计到达最后一行的所有路径长度（可以新建一个与triangle相同大小的list来存储），选出最短的即为最优路径
"""

class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        m = len(triangle)
        l = triangle   # l为新建的表格，存储矩阵中每一点的最优路径
        for i in range(1,m):  # 从第二行开始
            for j in range(i+1): # 索引为i的行有i+1列
                if j == 0: # 最左边
                    l[i][j] = l[i-1][0] + triangle[i][j]  # 这个地方改为 l[i][0]，也可
                elif j == i: # 最右边    注意这里是elif，否则可能执行两遍
                    l[i][j] = l[i-1][j-1] + triangle[i][j]
                else:
                    l[i][j] = min(l[i-1][j-1],l[i-1][j]) + triangle[i][j]
        return min(l[m-1][:])+1

# 60、n个骰子的点数（动态规划）
"""
扔 n 个骰子，向上面的数字之和为 S。给定 Given n，请列出所有可能的 S 值及其相应的概率。
n个骰子的所有点数的排列数为6^n，需要先统计出每个点数出现的次数，再除以6^n

https://www.cnblogs.com/bozhou/p/6971081.html
dp[i-1][j-1]存储着 扔第i个骰子时，此时扔出来的骰子的点数之和为j的 频数，dp是一个n行，6n列的矩阵
1 1 1 1 1 1 0 0 ... 0
0 1 3 4 5 6 1 0 ... 0
0 0 1 8=0+1+3+4.....0
dp[i][j] = dp[i - 1][j - 6] + dp[i - 1][j - 5] + dp[i - 1][j - 4] + dp[i - 1][j - 3] + dp[i - 1][j - 2] + dp[i - 1][j - 1]

"""
# coding=utf8
def get_ans(n):
    dp = [[0 for i in range(6 * n)] for i in range(n)]
    for i in range(6):
        dp[0][i] = 1
    # print dp
    for i in range(1, n):  # 1，相当于2个骰子。
        for j in range(i, 6 * (i + 1)):  # [0,i-1]的时候，频数为0（例如2个骰子不可能投出点数和为1），因此总是从i开始
            dp[i][j] = dp[i - 1][j - 6] + dp[i - 1][j - 5] + dp[i - 1][j - 4] + \
                       dp[i - 1][j - 3] + dp[i - 1][j - 2] + dp[i - 1][j - 1]
    count = dp[n - 1]
    return count  # 算得骰子投出每一个点数的频数。再除以总的排列数即可得到频率
print get_ans(3)  # 括号中的数字为骰子的个数。此代码为3个骰子时的情况。
# PS：python对下标为负值的数字会默认取零。所以省去了判断语句。


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

# 17、打印从1到最大的n位数 (最大的n位数即 pow(10,n)-1 或 写作10**n-1)
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

当模式中的第二个字符是*时： 
如果字符串第一个字符跟模式第一个字符不匹配，则字符串不变，模式后移2个字符，继续匹配(*使得前面的字符出现0次，即重新开始匹配)
如果字符串第一个字符跟模式第一个字符匹配，可以有2种匹配方式： 
1、字符串不变，模式后移2字符；（*使得前面的字符出现0次，即重新开始匹配） 
2、字符串后移1字符，模式不移动，即继续匹配字符下一位，因为*可以使得前面的字符出现任意次；

当模式中的第二个字符不是*时： 
1. 如果字符串第一个字符和模式中的第一个字符相匹配，那么字符串和模式都后移一个字符，然后匹配剩余的。 
2. 第一个字符不匹配，直接返回false。
"""
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        if s == pattern:
            return True
        if len(pattern) > 1 and pattern[1] == '*': # 第二个字符为*
            if s and (s[0] == pattern[0] or pattern[0] == '.'): # 第一个字符成功匹配：2种情况
                return self.match(s, pattern[2:]) or self.match(s[1:], pattern)
            else: # 第一个字符串未匹配或第二个字符不为*（合并这两种情况）
                return self.match(s, pattern[2:])
        elif s and pattern and (s[0] == pattern[0] or pattern[0] == '.'): # 第二个不为*，且第一个字符匹配
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
思路：用两个指针来实现：P1指向第一个数字的位置，P2指向最后一个数字的位置；P1只向后移动，P2只向前移动，
      while P1<P2：若P1指向的数字是偶数，P2指向的数字是奇数，则交换着两个数字；若P1指向的是偶数，P2指向的也是偶数，
      则P2-=1，P1不变；若P1指向的是奇数，P2指向的是奇数，P1+=1，P2不变；else：P1+=1,P2-=1:。
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
    def printMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if matrix == []:
            return []
        left, right, top, down = 0, len(matrix[0]) - 1, 0, len(matrix) - 1
        res = []
        while left < right and top < down:
            for i in xrange(left, right+1):
                res.append(matrix[top][i])
            top += 1
            for i in xrange(top, down+1):
                res.append(matrix[i][right])
            right -= 1
            for i in xrange(right, left-1, -1): # 起点，终点多走一步，1或-1
                res.append(matrix[down][i])
            down -= 1
            for i in xrange(down, top-1, -1):
                res.append(matrix[i][left])
            left += 1
        if top == down:  # 若最后只剩下最后一行数据
            for i in xrange(left, right+1):
                res.append(matrix[top][i])
            # res += matrix[top][left:right+1]
        elif left == right:  # 若只剩下一列数据
            for i in range(top,down+1):
                res.append(matrix[i][right])
        return res

# 顺时针填充方阵
# -*- coding:utf-8 -*-
class Solution(object):
    def generateMatrix(self, n):
        if not n:
            return []
        res = [[0 for _ in xrange(n)] for _ in xrange(n)]
        left, right, top, down, num = 0, n-1, 0, n-1, 1  # num为要填充到matrix中的第一个数字
        while left <= right and top <= down: # 若非方阵，则此处为while left<right and top<down，下面对只有一行或一列进行处理
            for i in xrange(left, right+1):
                res[top][i] = num
                num += 1
            top += 1
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


# 38、字符串的排列
"""
输入一个字符串,按字典序打印出该字符串中字符的所有排列（考虑顺序，长度为n）。
例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
"""
"""
思路：把字符串分成两部分：一部分是字符串的第i个字符，另一部分是第i个字符之外的所有字符（阴影区域）。
接下来求阴影区域的字符串的排列。
    def connect(self,ss,temp): # ss为供选择的字符list，temp用来存储本次得到的组合
        if ss=='':
            self.res.append(temp)
        else:
            for i in range(len(ss)):
                self.connect(ss[:i]+ss[i+1:],temp+ss[i])
        若提供的ss中包含重复的字符，则需要做一个 list(set(self.res))
"""
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        if len(ss) <= 0:
            return []
        self.res = [] # 用来存放最后的字符串
        self.connect(ss,'')
        # 由于可能会有重复的字符如aabc ，也就有重复的字符串，需要转为set，再转为list
        uniq = list(set(self.res))
        return sorted(uniq)    # 并按序排列
    def connect(self,ss,temp): # ss为供选择的字符list，temp用来存储本次得到的组合
        if ss=='':
            self.res.append(temp)
        else:
            for i in range(len(ss)):
                self.connect(ss[:i]+ss[i+1:],temp+ss[i])

"""
举一反三！！！
若面试题是按照一定的要求摆放若干个数字，则可以求出这些数字的所有排列，然后一一判断每个排列是不是满足题目给定的要求
例如：P219，正方体顶点放置数字问题：每个顶点上放置一个数字，对这8个数字的放置顶点位置进行排列，然后判断有没有某一个排列
符合题目给定的条件。
8皇后问题：定义一个8维向量，数组中第i个数字表示位于第i行的皇后的列号，将数组中的8个数字分别用0-7（表示他们所在的列）初始化
（题目要求皇后不能相互攻击，因此必不同列）；然后对这8个数字进行排列，查看是否有符合要求的排列：由于已经确定不同行列，
因此只需判断每一个排列对应的8个皇后是不是在同一条对角线上：i-j==a[i]-a[j] or j-i==a[i]-a[j].
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
        for i in xrange(1,2**n): # 这里产生n位的所有二进制数：如n=3，则产生001-111，也就是1-8
            temp = ''  # 用来存储本次的组合
            for j in xrange(n):
                # 判断i的第j位是否为1，若为1，则取出abc中第j位的字符，1<<j即1左移j位，产生：001,010,100等
                if i & 1<<j:
                    temp = temp + ss[j]
            L.append(temp)
        return L
c = Solution()
print c.Combination('abc')

78. Subsets  集合中为不同数字，输出所有可能的 数字组合
"""
Given a set of distinct integers, nums, return all possible subsets (the power set).
集合中为不同数字，输出所有可能的组合
Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

"""
class Solution(object):
    def subsets(self, nums):  # 43%，69ms
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = [[]]  # 中间变量法
        for num in nums:
            for temp in res[:]:
                # 这个地方一定要带切片啊，因为如果不带分片，就是一个浅拷贝，后面res有变化，导致res长度不断增加，不能退出for循环
                x = temp[:]  # x便是上轮res中的一项，注意要带切片！！！
                x.append(num)
                res.append(x)  # 第一轮大循环，x=[[],[1]],res=[[],[1]]；第二轮x为[2],[1,2]；第三轮x为3,13,23,123
        return res

90. Subsets II  含有重复数字求其所有 数字组合
"""
每次添加新的num时，加一句：if x not in res: 这样就不会出现两个 22 这种情况了。注意：后期去重不可取，因为2222这种无法通过
去重去掉。
"""
"""
Input: [1,2,2]
Output:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
"""
class Solution(object):
    def subsetsWithDup(self, nums):  # 43%，69ms
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums = sorted(nums)
        res = [[]]  # 中间变量法
        for num in nums:
            for temp in res[:]:
                x = temp[:]
                x.append(num)
                if x not in res:
                    # 加了这一句，防止出现相同list,注意 [1,4,4,4]和[4,4,1,4]被当作不同list，因此前面要先对nums排序！
                    # 否则按照我们的生成规则，nums=[4,4,4,1]时会生成这两种，且无法去重
                    res.append(x)  # 第一轮大循环，res=[[],[1]]；第二轮x依次为[2],[1,2],res为[ [],[1],[2],[1,2] ]
        return res

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


# 求数组：左边的数都小于等于它，右边的数都大于等于它
"""
利用一个辅助数组（rightMin），记录每一个元素右侧的最小值是多少（倒序遍历）。在顺序遍历时，保存当前最大值，即左边最大的。
若此时遍历到的数就是左边最大值且小于等于右边最小值，则这个数就是我们要找的数。
"""
# -*- coding:utf-8 -*-
class Solution:
    def solve(self, array):
        res = []
        rightMin = [float('inf')] * len(array)
        min = array[len(array)-1]
        for i in range(len(array)-1,-1,-1):
            if array[i] < min:
                min = array[i]
            rightMin[i] = min
        max = array[0]
        for i in range(len(array)):
            if array[i] >= max:
                max = array[i]
                if array[i] <= rightMin[i]:
                    res.append(array[i])
        return res

# 39、数组中出现次数超过一半的数字
"""
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

思路1（摩尔投票法）：数组中有一个数字出现的次数超过数组长度的一半，也就是说它出现的次数比其他数字出现的次数的和还要多。
在遍历数组的时候保存两个值，一个是数组中的一个数字，另一个是次数。当我们遍历到下一个数字的时候，
若下一个数字和我们之前保存的数字不同，则次数减一，若次数为0.我们需保存下一个数字，并将次数设为1；
若下一个数字和我们之前保存的数字相同，则次数加1.最终要找的数字肯定是最终保存着的数字（实际上也就是出现次数最多的数字）。
最后还要在遍历数组一次，统计这个数字的频次，若检查之后发现并没有达到数组长度的一半，则返回0。
这样便实现了O(n)的时间复杂度，O(1)的空间复杂度。

思路2：分区思想，寻找中位数：若存在出现次数超过一半的数字，那它必为中位数。找出中位数之和，再遍历一遍数组，看看是否符合。
"""
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        count = 0
        for i in numbers:  # now是此时保存着的数字
            if count == 0:
                now = i
            if now == i:
                count += 1
            else:
                count -= 1
        return now if numbers.count(now) > len(numbers)/2 else 0

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

# 数组中出现次数超过三分之一的所有数字
"""
首先我们肯定知道数组中出现次数超过⌊ n/3 ⌋次的最多有两个！因为如果3个的话，这三个数字的总次数 >  n，不可能的。
所以我们对这个题的做法同样使用摩尔投票法，先使用两个变量分别保存次数最多和次多的就可以了。
然后我们还需要再过一遍数组，判断次数最多和次多的是不是超过了 n/3 次，把超过的数字返回就行了。
"""
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        N = len(nums)
        m = n = cm = cn = 0  # m,n:此时出现最多的两个数字；cm，cn：统计此时m,n的出现次数
        for num in nums:
            if num == m:
                cm += 1
            elif num == n:
                cn += 1
            elif cm == 0: # 将此时的数字num作为新的m
                m = num
                cm = 1
            elif cn == 0: # 将此时的数字num作为新的n
                n = num
                cn = 1
            else:
                cm -= 1
                cn -= 1
        cm = cn = 0
        for num in nums:
            if num == m:
                cm += 1
            elif num == n:
                cn += 1
        res = []
        if cm > N / 3:
            res.append(m)
        if cn > N / 3:
            res.append(n)
        return res


# 40、最小的k个数（分区思想，寻找第k大的数）
"""
输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4.（注意k>n的情况,则输出[] ）
先排序，再取 tinput[:k].复杂度为O(nlogn)
实现复杂度为O(n)的思路：Partition，要找到数组第k大的数，这个数的左边(包含自己)就是最小的k个数。
先随机选择一个数字，然后调整数组中数字的顺序，使得比它小的都在左边，比它大的都在右边。若下标=k，则[:k]便是最小的k个数；
若它的下标大于k，那么第k大的数应该位于它的左边；若它的下标小于k，那么第k大的数位于它的右边，
则我们可以接着在它的右边部分的数组中查找。找到这个数之后，取这个数的左边(不包含自己)就是最小的k个数。
"""
# -*- coding:utf-8 -*-
l = [1, 9, 2, 4, 7, 6, 3]
min_k = find_least_k_nums(l, 6)
print min_k
def find_least_k_nums(alist, k):
    length = len(alist)
    #if length == k:
    #    return alist
    if not alist or k <=0 or k > length:
        return
    start = 0
    end = length - 1
    index = partition(alist, start, end) # 分区操作：即比base小的放左边，比base大的放右边；返回基准值在分区后的位置（索引）
    while index != k:
        if index > k: # 要寻找的一定在左边，相当于给我一个新的alist，重新对此alist进行分区。注意这里alist始终同样大小长度
            index = partition(alist, start, index - 1)        # 只不过分区的区域改变了。因此最终的索引依然是alist中的索引。
        elif index < k:
            index = partition(alist, index + 1, end)
    return alist[:k]
def partition(alist, start, end):
    if end <= start:
        return
    base = alist[start]
    index1, index2 = start, end
    while start < end:
        while start < end and alist[end] >= base: # 从后往前，利用while循环找到小于base的元素位置end
            end -= 1
        alist[start] = alist[end] # 将小于base的此元素放到start位置上（此时end处假定为空，可以放其他的元素）
        while start < end and alist[start] <= base: # 从前往后，找到大于base的元素位置start，并将此元素放到end位置上
            start += 1
        alist[end] = alist[start]
    alist[start] = base # 将此时的基准值放在此时start位置
    return start


class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if tinput is None:
            return
        if k > len(tinput):
            return []
        tinput.sort()
        return tinput[:k]

# 42、连续子数组的最大和
"""
例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。
给一个数组，返回它的最大连续子序列的和.(子向量的长度至少是1)
注意：可以从中间开始连续！
思路：记录从第一项开始的和sum，若sum<0,则置为0，然后继续累加，max存储此过程中sum的最大值。考虑array全为负数的情况，需要记录
array中的最大值max1，当max=0时，返回max1，max大于0时，返回max。这样的空间复杂度为O(1)。
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

"""
可以当作是一个简单的动态规划问题：
局部最大值temp_max代表以当前位置为结尾的最大连续子串的最大值（注意！！是以当前位置为结尾！！）
此时的局部最大值 = max(之前的局部最大值+结尾位置值 ，结尾位置值)
而此时连续数组的最大值 = 所有出现过的temp_max中的最大值，max(temp_max,res)
temp_max = max(temp_max+nums[i] ,nums[i])
res = max(temp_max,res)
这样的空间复杂度为O(n)。
"""
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return nums[0]
        temp_max = nums[0]
        res = nums[0]
        for i in range(1,len(nums)):
            temp_max = max(temp_max+nums[i] ,nums[i])
            res = max(temp_max,res)
        return res

152. Maximum Product Subarray 连续子数组的最大乘积 （动态规划）
"""
Input: [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
思路： 
某个位置可能出现了0或者负数，遇到0时，整个乘积会变成0；遇到负数时，当前的最大乘积会变成最小乘积，最小乘积会变成最大乘积
使用两个数组分别记录 以某个位置i结尾 的最大乘积和最小乘积了。令最大乘积为f，最小乘积为g。那么有：

当前的最大值等于 已知的最大值、最小值分别与当前值的乘积、当前值，这三个数的最大值。
当前的最小值等于 已知的最大值、最小值分别与当前值的乘积、当前值，这三个数的最小值。
结果是得到的f数组中的最大值。
注意这里是记录 以某个位置i结尾的 ；因为题目要求是连续子数组。
"""
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        n = len(nums)
        f = [0]*n  # 不能简单定义为 []
        g = [0]*n
        f[0] = g[0] = res_max = nums[0]
        for i in range(1,n):
            f[i] = max(f[i - 1] * nums[i], g[i - 1] * nums[i], nums[i])
            g[i] = min(f[i - 1] * nums[i], g[i - 1] * nums[i], nums[i])
            res_max = max(f[i],res_max)
        return res_max

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
        for i in xrange(n):       # 生成数字序列0,1,2,3.。。，10,11,12。。。(n>10时第n位肯定没到第n个数字)
            sum += len(str(i))   # 一直从0累加到数字i的位数之和
            if sum >= n: # 说明i的某一位就是我们要寻找的第n位对应的数字
                # 如果sum刚好等于n，则就是i的第一位，即i[0],0=sum-n，应返回 i[sum-n]
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

# 48、最长不含重复字符的子字符串
"""
给定一个字符串，请找出其中无重复字符的最长子字符串。
例如，在”abcabcbb”中，其无重复字符的最长子字符串是”abc”，其长度为 3。 
对于，”bbbbb”，其无重复字符的最长子字符串为”b”，长度为1。
"""
"""
思路：遍历字符串中的每一个元素。
用一个整形变量start存储当前无重复字符的子串开始的下标。
借助一个dict来存储某个元素最后一次出现的下标，
若从某start开始的子串中出现了重复字符(意味着在此时的序列中)！start更新为重复的那个位置+1，并更新此元素出现的坐标
记录下此时序列长度，用max1来存储每次得到的序列的长度的最大值。
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
            if s[i] in d and d[s[i]] >= start:
                start = d[s[i]] + 1   # start更新为重复的那个位置+1，相当于把之前那个被重复的字符以及它之前的序列全移出去
            d[s[i]] = i # 更新此元素本次出现的坐标
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
即丑数应该是另一个丑数乘以2、3或5得到的结果。
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
        # 说人话：index2就是丑数列表中由某一个丑数乘以2得到的最新的索引，index2记录着所有通过乘以2得到的丑数。
        index2 = 0
        index3 = 0
        index5 = 0
        for i in range(index-1):
            newUgly = min(uglylist[index2]*2 ,uglylist[index3]*3 ,uglylist[index5]*5)
            uglylist.append(newUgly)
            # 看看这个丑数是通过乘以几得到的，更新基准值
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
# 在d中第二次遍历，扫到的第一个value为1的。
# 如果觉得dict消耗的空间太大，可以使用数组
由于字符（char）是长度为8的数据类型，共有256中可能，因此哈希表可以用一个长度为256的数组来代替，(意思就是说所有的字符个数不超过256个)
数组的下标相当于键值key，对应字符的ASCII码值；数组的值相当于哈希表的值value，用于存放对应字符出现的次数。
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
        # for i in data: # 相当于实现了一个对data的深复制
        #     copy.append(i)
        copy = data[:] # 相当于实现了一个对data的深复制
        copy.sort()

        for i in range(len(copy)): # 记录下有序序列中索引与原始序列索引的差值，注意每次记录一个差值之后要remove此数
    # 由于每次都要remove此数，所以实际上每次比较的时候，copy[i]在copy中的索引都是0(对于已排序copy不必执行remove，因为其索引已知为0)
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
            if mid>0 and data[mid-1]!=k or mid==0: # 是第一个k
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
            if mid<len(data)-1 and data[mid+1]!=k or mid==len(data)-1:
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
# class Solution:
#     def GetNumberOfK(self, data, k):
#         # write code here
#         if not data:
#             return 0
#         if len(data) == 1 and data[0] != k:
#             return 0
#         left = 0
#         right = len(data) - 1
#         first_k = 0
#         while left <= right:
#             mid = (left + right) // 2
#             if data[mid] < k:
#                 left = mid + 1
#             elif data[mid] > k:
#                 right = mid - 1
#             else:
#                 if mid == 0:
#                     first_k = 0
#                     break;
#                 elif data[mid-1] != k:
#                     first_k = mid
#                     break;
#                 else:
#                     right = mid - 1
#         left = 0
#         right = len(data) - 1
#         last_k = -1
#         while left <= right:
#             mid = (left + right) // 2
#             if data[mid] < k:
#                 left = mid + 1
#             elif data[mid] > k:
#                 right = mid - 1
#             else:
#                 if mid == len(data) - 1:
#                     last_k = len(data) - 1
#                     break;
#                 elif data[mid+1] != k:
#                     last_k = mid
#                     break;
#                 else:
#                     left = mid + 1
#         return last_k - first_k + 1

"""
在范围0~n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字

利用递增：假设m不在数组中，那么m+1就处在下标为m的位置，可以发现：m正是数组中第一个值和下标不相等的元素的位置
找到某一个值和下标不相等的元素后，向前继续探索，直到找到第一个二者不相等的元素 def get_first(self,data,k,start,end):
"""
# 二分法实现:
while start<=end:
    mid = (start+end)/2
    if data[mid]==mid:
        start = mid+1  # 在右边寻找
    elif data[mid-1]==mid-1 or mid==0:
        return mid  # 找到了
    else:
        end = mid -1 #

"""
单调递增数组里的每个元素都是整数且唯一，找出数组中任意一个数值等于其下标的元素
那么:前面的必定<=其下标，后面的>=其下标
"""
#二分法：
if data[mid]==i:
    return data[mid]
if data[i]>i :
    end = mid-1
else :
    start=mid+1


# 56、数组中数字出现的个数
"""
题目1：在一个数组中除了一个数字只出现一次之外，其他数字都出现了2次，请找出那个只出现了一次的数字。
要求：线性时间复杂度O(n)，空间复杂度为O(1)
思路：用位运算来解决XOR异或来解决该问题。由于两个相同的数字的异或结果是0，我们可以把数组中的所有数字进行异或操作，
结果就是唯一出现的那个数字。
比如：依次取出数字12233的各位即1 2 2 3 3，也就是0001 0010 0010 0011 0011，对这五个数字做一个异或，
可以改变 异或 的顺序，因为2和2,3和3终将进行异或计算，最终得到0000,1与0做异或还是本身。所以最终只留下0001，即1。
同理适用于：
数组中只出现一次的数字：一个整型数组里除了一个数字之外，其他的数字都出现了偶数次。请写程序找出这个只出现一次的数字。

def singleNumber(nums):
    ans = 0
    for i in range(len(nums)):
        ans ^= nums[i]
    return ans

题目2：数组中出现一次的两个数字（medium）
在一个数组中除了2个数字只出现一次之外，其他数字都出现了2次，请找出两个只出现了一次的数字。

思路：从头到尾异或数组中的每个数字，可以得到只出现1次的两个数字的异或结果。
从异或结果中，找到右边开始第一个不为0的位数（利用右移位的指令+与操作），记为n。表明这两个数字的第n位不相同。
我们将数组中所有的数字，按照第n位是否为0，分为两个数组。
每个子数组中都包含一个只出现一次的数字，其它数字都是两两出现在各个子数组中。那么结合题目一，我们已经得出了答案。
 
 def TwoNumber(nums):
    ans = 0
    for i in range(len(nums)):
        ans ^= nums[i]
    flag = 1
    while ans & flag == 0:
        flag <<= 1
    nums1 = 0 # 答案写的是 nums1 = ans
    nums2 = 0
    for i in range(len(nums)):
        if data[i]&flag == 0:
            nums1 ^= data[i]
        else:
            nums2 ^= data[i]
    return nums1,nums2
 
题目3：在一个数组中除了一个数字只出现一次之外，其他数字都出现了3次，请找出那个只出现了一次的数字。

int 整型共有32位，用bitnum[32]存储这n个数据每个二进制位上1出现的个数，再%3，
如果为1，那说明这一位是要找元素二进制表示中为1 的那一位。

def singleNumber1(nums):
    ans = 0
    for i in range(1,33):  # i是某一个数字的第一位到第32位  （答案上写的是 从0-31）
        bit = 0
        for j in range(n):  # 将所有这n个数字中第i位拿出来累加，存储到bit中
            bit += (nums[j]>>i)&1 # 将第i位上的数字（1或0）累加
        ans |= (bit%3)<<i # 将ans第i位上的值设为(bit%3)
    return ans

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
有序序列中输出所有和为S的连续正数序列。
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
        sum = 3  # 序列所有值累加到这，改变start就减去start，改变end就加上end
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
                n += new_list[j + 1] - new_list[j] -1  # 统计空缺值，比如6-5-1=0，即他们之间的空缺牌个数为0.
            else: # 若出现==0的情况，说明有重复的牌，返回False
                return False
        if n <= m :
            return True
        else:
            return False


# 62、圆圈中最后剩下的数字（约瑟夫环问题）
"""
0,1,...,n-1这n个数字排成一个圆圈，从0开始数，每次删除第m个数字（从目前存在的地方向前走m-1步，删除之）。
如 0,1,2,3,4。从0开始每次删除第三个数字，则删除的前4个数字依次为2,0,4,1，因此最后剩下的数字是3
两种解法：第一种构造一个环形链表，每次在这个链表中删除第m个节点
第二种：找规律,设被删去的位置为i，每次从上次删除的点之后（已执行res.pop(i)，因此此时res[i]对应的就是上次删除之后的点的下一个点）
出发向前走m-1步，新的起点i就等于 (i + m - 1) % len(res)，然后执行 res.pop(i)，即删除这个点i，使得下次的起点为被删除点的下一个点
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
前面i-1个数字时的最小值即可。不断更新此时的最大值。
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

122. Best Time to Buy and Sell Stock II  卖股票
"""
Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
可以多次买入卖出。
正确做法：采用贪心算法，只有当前比前一天大，就卖股票（如果知道后面有比今日大的，又买回来），这样等于上面的操作
"""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices or len(prices) == 1:
            return 0
        maxpro = 0
        for i in range(1,len(prices)):
            if prices[i] > prices[i-1]:
                maxpro += prices[i] - prices[i-1]
        return maxpro

123. Best Time to Buy and Sell Stock III 卖股票3
"""
Input: [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
             Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
You may complete at most two transactions.最多只有两次操作：即 买卖买卖，且买和卖不能在同一天。     
https://www.jianshu.com/p/e939093a20bb  动态规划    
"""
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        len_prices = len(prices)
        pay_before, profit_before, pay_after, profit_after = -prices[0], 0, -prices[0], 0
        for i in range(1, len_prices):
            pay_before = max(pay_before, -prices[i])  # 第一次购买时候的最低价格，买入价格（是一个负数）,这时候是你的收益为负数
            profit_before = max(profit_before, pay_before + prices[i])  # 第一次以这个价格卖出时获得的最大收益，这是在你这个价格前买后卖的收益
            pay_after = max(pay_after, profit_before - prices[i])  # 第二次购买之后的收益（这个是你用第一次的收益买了之后的最大收益,或者是前面的更高的收益）
            profit_after = max(profit_after, pay_after + prices[i])  # 第二次以这个价格卖出所获得的最大收益（已经囊括了第一次收益）
        return profit_after


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
其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]，B[0] = 1。不能使用除法。
思路：将它分为A[0]*...*A[i-1],以及A[i+1]*...*A[n-1]。注意：当A长度为1时即为0时，取B[0]=1。
"""

# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        # write code here
        if not A:
            return []
        # 计算序号在B[i]之前的，如i=3，乘上A[0],A[1],A[2]
        num = len(A)
        B = [None] * num  # 否则后面B[i-1]会报错
        B[0] = 1
        # 前半部分：
        # B[1] = A[0]
        # B[2] = A[0]*A[1]
        # B[n] = ...*A[n-1]
        # 即 B[i] = B[i - 1] * A[i - 1]
        for i in range(1, num):
            B[i] = B[i - 1] * A[i - 1]
        # 后半部分
        # 从倒数第二项开始，因为B[n-1]项已经乘完了
        # B[n-2] *= A[n-1]
        # B[n-3] *= A[n-1]*A[n-2]
        # 用tmp来做乘积，A[n-1]*A[n-2]累乘
        # 即 tmp *= A[i + 1]
        #    B[i] *= tmp
        tmp = 1
        for i in range(num - 2, -1, -1):
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
        except:
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
def quick_sort(arr):
    '''''
    模拟栈操作实现非递归的快速排序
    '''
    if len(arr) < 2:
        return arr
    stack = []
    stack.append(len(arr)-1)
    stack.append(0)
    while stack:
        l = stack.pop()
        r = stack.pop()
        index = partition(arr, l, r)
        # 分别对基准值左边以及右边的序列加入栈（接着出栈做分区操作）
        # 注意判断是否还需要做分区（保证分区的长度至少为2，即索引之差至少为2）
        if index - l >= 2:
            stack.append(index - 1)
            stack.append(l)
        if r - index >= 2:
            stack.append(r)
            stack.append(index + 1)
def partition(arr, start, end):
    # 分区操作，返回基准线下标
    pivot = arr[start] # 基准值定为第一个值
    while start < end:
        # 从后向前，找到比基准值小的元素，并放到start位置（即基准值的初始化位置）
        while start < end and arr[end] >= pivot:
            end -= 1
        arr[start] = arr[end]
        # 从前向后，找到比基准值大的元素，并放到end位置（该位置上的值已经放到start位置上去了）
        while start < end and arr[start] <= pivot:
            start += 1
        arr[end] = arr[start]
    # 此时start = end，基准值就放在这个位置，并返回基准值位置
    arr[start] = pivot
    return start


# 并查集
"""
例如给定好友关系为：[0,1], [0, 4], [1, 2], [1, 3], [5, 6], [6, 7], [7, 5], [8, 9]。
在这些朋友关系中，存在3个朋友圈，分别是【0,1,2,3,4】，【5,6,7】，【8,9】
下面用数组替代了dict来存储各个节点的父节点（索引下标为节点，值为该节点的父节点），这样至少能比dict节约一半的空间消耗。
# 也可以用字典来实现：https://www.cnblogs.com/lateink/p/6437439.html  or  https://www.cnblogs.com/jiaxin359/p/9265208.html
"""

def union_find(nodes, edges):
    father = [0] * len(nodes)  # 记录父节点
    for node in nodes:  # 初始化为本身
        father[node] = node
    for edge in edges:  # 标记父节点
        head = edge[0]
        tail = edge[1]
        father[tail] = head
    for node in nodes:
        while True:  # 循环，直到找到根节点
            father_of_node = father[node] # 取出它此时的父节点
            if father_of_node != father[father_of_node]:
                father[node] = father[father_of_node]
            else:  # 如果该节点的父节点与其父节点的父节点相同，则说明找到了根节点
                break
    L = {}  # 放到一个字典中去
    for i, f in enumerate(father):
        L[f] = []
    for i, f in enumerate(father):
        L[f].append(i)
    return L

nodes = list(range(0, 10))
test_edges = [[0, 1], [0, 4], [1, 2], [1, 3], [5, 6], [6, 7], [7, 5], [8, 9]]
L = union_find(nodes, test_edges)
print(L)
print('num of pyq:', len(L))


# 华为实习笔试

"""
华为19.3
abc3(ABC)
CBACBACBAcba
"""
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
test = [[0 for _ in range(n)] for _ in range(m)]
def dfs(i,j,end_x,end_y):
    if i==end_x and j==end_y: # 有一条路径到达了终点，count+1
        C.count += 1
        return # 立即返回，终点始终不会标记为1！
    test[i][j]=1 # 走过的格子标记为1
    for r,c in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
        if r>=0 and r< n and c>=0 and c< m and test[r][c]==0 and state[r][c]>state[i][j]:
            # 由于每次的海拔要比之前高，所以和13题先判断是否要return不同，这里写的是进入dfs的条件
            dfs(r,c)
    test[r][c]=0
dfs(start_x,start_y,end_x,end_y)
print C.count
https://leetcode-cn.com/tag/array/ 可以看题目对应的中文名字是什么

1. Two Sum == target 两数相加为目标值

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 注意demo给的是有序列表，实际上是有可能出现无序列表的
        # 由于要求的两数之和等于targe，因此遍历数组，查看接下来是否会出现target-当前元素 的值，用一个字典来存储
        # key：当前数字需要配对为target的数字，value：当前数字的index
        nums_dict = {}
        for i, num in enumerate(nums):
            if num in nums_dict: # 当合适的数出现，返回和他对应的数的索引，他自己的索引
                return [nums_dict[num], i]
            else:
                nums_dict[target - num] = i

15. 3Sum == 0   三个数相加为0

#-*- coding:utf-8 -*-
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        """
        思路：先排序。先固定一个数，然后再找某两个数，使得三个数之和为0.可以推广到 k-sum ，先做 k-2次循环，在最内层做夹逼
         如果要求按照原来的顺序输出字符，可以将result.sort(key=nums.index)。
         注意：固定一个数之后，剩下的两个数只需要在它后面的数里面去找，因为这是个组合问题，就算前面有匹配的，那必然
         在之前的工作中已经找到了。还有一点启发是，先实现比较特殊的情况，说不定这就是所有的case了。
        """
        result = []
        if len(nums) < 3:
            return result
        nums.sort()  # 先排序
        for i in range(0, len(nums)):
            left = i + 1 #
            right = len(nums) - 1
            # 注意跳过重复的数！后面也有类似的处理
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum == 0:
                    result.append((nums[i], nums[left], nums[right]))
                    # 注意：此时应该同时移动left和right，因为此时绝对不可能出现left和right前一个匹配的情况
                    # 匹配上那也是重复的一对组合
                    left += 1
                    right -= 1
                    # 若出现大量的重复情况，改变left和right的一个就好了
                    while nums[left] == nums[left - 1] and left < right: # 因为此时left+right绝对不可能为sum或者与之前sum相同
                        left += 1
                    while nums[right] == nums[right + 1] and left < right:
                        right -= 1
                elif sum > 0:
                    right -= 1
                    while (nums[right] == nums[right + 1] and left < right):
                        right -= 1
                else:
                    left += 1
                    while (nums[left] == nums[left - 1] and left < right):
                        left += 1
        return list(set(result))  # 去重

16. 3Sum Closest==target  三个数相加最接近目标值

...
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum == target:
                    return sum   # 直接返回，不必再寻找其他的
                elif sum > target:
                    if abs(sum-target) < abs(result-target):  # 记录最接近的值
                        result = sum
                    right -= 1
                    while (nums[right] == nums[right + 1] and left < right):
                        right -= 1
                else:
                    if abs(sum-target) < abs(result-target):
                        result = sum
                    left += 1
                    while (nums[left] == nums[left - 1] and left < right):
                        left += 1
        return result

18. 4Sum == target  四个数相加为目标值

...
        nums.sort()  # 先排序
        for k in range(len(nums)):
            if k > 0 and nums[k] == nums[k - 1]:
                continue
            for i in range(k+1, len(nums)):
                if i > k+1 and nums[i] == nums[i - 1]:
                    continue
                left = i + 1  #
                right = len(nums) - 1
                while left < right:
                    sum =  nums[k] + nums[i] + nums[left] + nums[right]
                    if sum == target:
                        ...

11. Container With Most Water  放置最多水的容器，选出此时木板的位置（木板高度不一）
"""
i和j分别指向前后的两个端点，因为木桶原理，容积取决于行长度和最短高度的积，所以，两个端点高度较低的需要移动，
因为高度较高的移动不可能大于原来的两端点积。这样，每次都是高度低的移动，直到两指针相邻。
"""

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        i = 0
        j = len(height)-1
        water = (j - i) * min(height[i], height[j])
        best = [i,j]
        while i<j:
            if height[i]<=height[j]:
                i += 1
            else:
                j -= 1
            temp = (j - i) * min(height[i], height[j])
            if temp>water:
                water = temp
        return water


26. Remove Duplicates from Sorted Array 排序列表去重

# 返回去重之后的列表长度以及列表：实现空间复杂度为1的操作
# 将每个第一次出现的元素移到j所在的位置的后一位，同时将j加1（相当于用原列表的前部分存储所有不重复的元素）
# 最后得到j+1个非重复的元素，排列在原来序列的前j+1个位置（即index为0-j）

class Solution(object):
    def removeDuplicates(self, nums):
        if len(nums)==0:
            return 0
        j=0
        for i in range(1,len(nums)): # 从第二个位置开始，第一个元素作为第一个不重复的元素，放在j=0的位置
            if nums[i] != nums[j]: # 若此时遍历到的位置i上的元素与此时j位置元素不同，则放在j的下一个位置，j+=1
                nums[j+1] = nums[i]
                j = j+1
        return  j+1,nums[:j+1]
c = Solution()
print c.removeDuplicates([1,1,2,4,5,6,6,6,7,7])

80. Remove Duplicates from Sorted Array II  排序列表去重2（允许有两个重复的元素）

"""
Given nums = [1,1,1,2,2,3],
Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
只允许两个重复的元素，多的去掉。
使用两个指针prev和curr，判断A[curr]是否和A[prev]、A[prev-1]相等，如果相等curr指针继续向后遍历，
直到不相等时，将curr指针指向的值放到prev+1的位置，同时将prev和curr向后移一位

还有一个方法：类似于基数排序，统计各个元素个数，大于2将dict中的value改为2，小于2保留原value
"""

class Solution(object):
    def removeDuplicates(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(A)<=2:
            return len(A)
        prev = 1 # 存储着此时不重复的元素
        curr = 2 # 从第三个开始
        while curr <= len(A)-1:
            if A[curr] == A[prev] and A[curr] == A[prev-1]:
                curr += 1   # 此时返回prev+1，始终为2个数
            else:
                A[prev+1] = A[curr]
                prev += 1
                curr += 1
        return prev+1

27. Remove Element 删除给定值val并返回数组长度
# j初始化为0，只要你不是val，那么我就把你放到j的位置，同时j向后移动一步，和前面几道题类似，实现了O(1)的空间复杂度

class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        j = 0
        for i in range(0,len(nums)): # 从第一个元素开始，不是val的就放在此时j位置上
            if nums[i] != val:
                nums[j] = nums[i]
                j = j+1
        return j

31. Next Permutation  输出字典序的下一个排列
"""
输出字典序中的下一个排列。比如123生成的全排列是：123，132，213，231，312，321。那么321的next permutation是123。
"""
# -*- coding:utf-8 -*-
# https://blog.csdn.net/qq_28119401/article/details/52972616 忘记原理了（用提取规则的方法来实现）
class Solution(object):
    def nextPermutation(self, nums):
        if len(nums)<=1:
            return
        for i in range(len(nums)-2,-1,-1):
            if nums[i]<nums[i+1]:
                for k in range(len(nums)-1,i,-1):
                    if nums[k]>nums[i]:
                        nums[i],nums[k]=nums[k],nums[i]
                        nums[i+1:]=sorted(nums[i+1:])
                        break
                break
            else:
                if i==0:
                    nums.sort()


class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        nums.sort()
        temp = map(str,nums) # 得到一个包含所有字符的list
        temp = ''.join(temp)
        self.L = list() # 用来存放最后的字符串
        path = ''  # path是已排好的组合
        self.connect(temp,path) # ss是剩下的待排序的字符

        uniq = L  # 只要传入的list是有序的，那么排列而成的结果便也是有序的
        print uniq
        for i in range(len(uniq)):
            if uniq[i] == temp and i+1<len(uniq):
                temp1 = map(int,[j for j in uniq[i+1]])
                return temp1
            else:
                temp1 = map(int, [j for j in uniq[0]])
                return temp1

    def connect(self,ss,path):  # ss为供选择的字符，L
        if ss=='':
            self.L.append(path)
        else:
            for i in range(len(ss)): # 每次拿出一个字符，然后将剩下字符的组合+此时拿出的字符
                self.connect(ss[:i]+ss[i+1:],path+ss[i])
c = Solution()
print c.nextPermutation([1,2,3])


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


40. Combination Sum II
"""
在39的基础上，增加一项条件，每个数字最多只能使用一次。但是同时也会存在重复的数字。
Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
"""
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.res = []
        candidates = sorted(candidates)
        used = [0]*len(candidates) # 用过的数字就标记为1
        self.dfs(candidates,[],target,used,0)
        return self.res

    def dfs(self,candidates,sub,target,used,last):
        if target == 0:  # 目标变为0，说明已凑好一组
            self.res.append(sub[:])
        if target < candidates[0]: # 目标不为0且小于此时最小值，说明此组无法凑成
            return
        temp = None # 为了防止重复的比如两个1，那在一层递归只处理一次
        for i in range(len(candidates)):
            n = candidates[i]
            if n > target:
              return
            if n < last or used[i] == 1 or temp == n:
              # used矩阵用来保证每一次大循环不重复选择已被选择的元素，temp使得每一层递归只处理一次
              # 如果在这一层递归的时候 比如有两个1， 那之前做一次1的时候，第二次就不处理了，不然会重复
              continue
            sub.append(n)
            used[i] = 1  # 记录是否 用过的
            self.dfs(candidates, sub, target - n, used, n)
            temp = n  # 注意：这个是放在dfs之后的
            used[i] = 0
            sub.pop()

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


48. Rotate Image  把正方形矩阵原地顺时针旋转90度。
"""
Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],
rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
方法：先上下翻转：
7 8 9
4 5 6
1 2 3
再沿对角线翻转：
7 4 1
8 5 2
9 6 3
"""
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]  这种形式也能直接用matrix[i][j]来操作
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        # 做上下翻转（逐行对所有的列做处理，只处理前 rows/2+1 行）
        for i in xrange(rows/2):
            for j in xrange(cols): # 0对应2，第i行对应的行为rows-1-i，rows为整数，rows/2得到的也是整数
                matrix[i][j],matrix[rows-1-i][j] = matrix[rows-1-i][j],matrix[i][j]
        # 做对角线翻转（逐行对前i列做处理，第二行对8进行处理，第三行对9、6进行处理）
        for i in xrange(rows):
            for j in xrange(i): # 注意！这里只对前i列做翻转处理，否则后面会重复处理
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]

55. Jump Game 跳格子，列表中每一项代表这一次能跳的最大长度，是否能跳到终点
"""
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
利用贪心法，遍历从倒数第二个到第一个的所有点，寻找离终点最近的且能到达终点的位置，将其设为新的终点，直到循环结束，若此时
终点==起点，说明从起点能够到达终点。
"""
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) < 2:
            return True
        tmp = len(nums)-1
        for i in range(len(nums)-2,-1,-1): # 从倒数第二个到第一个
            if nums[i] >= tmp-i: # 若从倒数第二个能走到最后一个，那么终点位置定为倒数第二个，继续寻找一个能走到终点的位置(贪心法)
                tmp = i          # 再将此位置定为新的终点，不断重复，直到遍历完所有的点
        if tmp == 0:             # 若循环结束之后，终点位置为起点，则说明能到达
            return True
        return False
# 最短的路径长度是多少？（hard）
"""
记录在n步能走的最远距离maxpos，当超越最远距离则，n=n+1，依此循环即可求解，具体代码如下，下面讲一下具体逻辑

（1）step记录步数

（2）lastpos记录step步数时，能走的最远距离

（3）maxpos记录当前位置i能走的最远距离，为原maxpos和i+nums[i]的最大值

（4）当位置i超越step能走的lastpos距离时，则需要增加一步step+=1，并更新此步能走的最远距离lastpos=maxpos
"""
def jump(nums):
    if len(nums) < 2:
        return 1
    step = 0
    lastpos = 0
    maxpos = 0
    for i in range(len(nums)):
        if i > lastpos:
            lastpos = maxpos
            step += 1
        maxpos = max(maxpos,i+nums[i])
    return step

56. Merge Intervals 合并区间
"""
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
（首先按照起点排序 sorted(intervals,key=lambda x:x[0])）
"""

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if not intervals:
            return []
        if len(intervals) == 1:
            return intervals  # [][0]会报错
        # 首先按照起点位置排序
        intervals = sorted(intervals,key=lambda x:x[0])
        # 排好序之后判断下一个区间的起点是否在前一个区间中，若在则合并；若不在，则将新区间添加。
        res = [intervals[0]] # 初始值:仅包含第一个区间，通过改变此时res中最后一个区间的末尾来执行合并
        pre = res[0] # 取得前一个区间
        for elem in intervals[1:]:
            if pre[1] >= elem[0]: # 合并
                if pre[1] <= elem[1]:
                    pre[1] = elem[1] # 将此时res中的最后一项（pre区间）的终点设为elem的终点（合并区间）
                # 若 pre[1]>elem[1]，不必做任何操作
            else: # 毫无交集的情况，将此时的区间加到res中（前一个区间已经在res中）
                res.append(elem)
                pre = res[-1]  # 此时pre取得res中的最后一项
        return res

66. Plus One
"""
Input: [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
"""

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        s = map(str,digits)  # python3的map函数返回的是一个迭代器，list(map(str,digits))这样才能收获一个list
        temp = str(int(''.join(s))+1)
        res = []
        for i in range(len(temp)):
            res.append(int(temp[i]))
        return res

73. Set Matrix Zeroes 设置行列为0
"""
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.（空间复杂度为1）
Input: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
Output: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
"""
# 空间复杂度为 O(m+n)的实现，时间复杂度为 O(m*n)
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        rows, cols = set(), set()
        # 记录需要变为0的行和列
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)
        # 将上述需要变为0的行或列对应的矩阵处变为0
        for i in range(m):
            for j in range(n):
                if i in rows or j in cols:
                    matrix[i][j] = 0
# 空间复杂度为O(1)的实现,同时保持O(m*n)的时间复杂度
class Solution:
    # @param matrix, a list of lists of integers
    # RETURN NOTHING, MODIFY matrix IN PLACE.
    def setZeroes(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        row = False     # 用于判断第一列是否存在0
        column = False  # 用于判断第一行是否存在0

        for i in range(m):
            if matrix[i][0] == 0:
                row = True
                break
        for j in range(n):
            if matrix[0][j] == 0:
                column = True
                break
        # 对内部元素进行判断，直接将存在0的元素的行列索引存储到边缘两边，虽然改变了边缘的值，但之前已经判断过了，因此无关紧要
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        # 对内部元素进行赋值0
        for i in range(1, m):
            if matrix[i][0] == 0:
                for j in range(1, n):
                    matrix[i][j] = 0
        for j in range(1, n):
            if matrix[0][j] == 0:
                for i in range(1, m):
                    matrix[i][j] = 0
        # 对边缘元素进行赋值
        if row:
            for i in range(0, m):
                matrix[i][0] = 0
        if column:
            for j in range(0, n):
                matrix[0][j] = 0

74. Search a 2D Matrix  在二维矩阵中寻找目标值
"""
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
Output: true
按大小排序的二维矩阵，判断目标值是否存在于矩阵中
思路：先与每一行最后一个数比较，找出目标所在的行数，若没找到，直接返回false
      再在此行寻找
"""

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        temp = matrix   # 不能直接做 matrix == []这样的判断，会显示非全局变量
        if temp == [] or temp == [[]]:  # leetcode卡得好严格
            return False
        m = len(matrix)
        n = len(matrix[0])
        row = 0
        for i in range(m): # 确定target所在的行
            if target > matrix[i][n-1]:
                row = i + 1
            elif target == matrix[i][n-1]:
                return True
            else:
                break
        if row >= m:
            return False

        j = 0
        while j<n:  # 确定target在这一行中对应的列
            if matrix[row][j] == target:
                return True
            elif matrix[row][j] < target:
                j += 1
            else:
                return False

75. Sort Colors 对颜色进行排序
"""
Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
实现原地排序
Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
"""
# 计数排序：线性复杂度，常数空间复杂度
# 因为只有三个数，所以简单的方法是计数排序。第一次遍历，统计出这三个数字出现的次数，
# 第二次遍历，根据三个数字的次数对原列表进行修改。
from collections import Counter
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        count = Counter(nums)
        for i in xrange(len(nums)):
            if i < count[0]:
                nums[i] = 0
            elif i < count[0] + count[1]:
                nums[i] = 1
            else:
                nums[i] = 2

# 思路2：快排之三项切分快速排序
"""
如果只能扫一遍，很容易想到的就是左边存放0和1，右边存放2.两边往中间靠。
设置两个指针，zero和two；zero指向第一个1的位置（0串的末尾），two指向第一个非2的位置。然后用i对nums进行遍历：
然后使用i从头到尾扫一遍，直到与two相遇。
i遇到0就换到左边去，遇到2就换到右边去，遇到1就跳过。
"""
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        zero = 0
        two = len(nums) - 1
        i = 0
        while i <= two:
            if nums[i] == 0:   # 0放到此时最前面的位置去，并将zero指针向后移一位，i继续向后走
                nums[zero], nums[i] = nums[i], nums[zero]
                i += 1
                zero += 1
            elif nums[i] == 1:  # 不改变1的位置，i继续向后走
                i += 1
            elif nums[i] == 2:  # 将2放到最后去，并将two指针前移一位
                nums[two], nums[i] = nums[i], nums[two]
                two -= 1

118. Pascal's Triangle  生成杨辉三角形
"""
Input: 5
Output:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
思路：
假设rowlist是每一层的数组，rowlist第i个值为上一层数组的第i和i+1之和 
设定一个临时数组，为上一层数组首尾各加上一个0，rowlist第i个值为临时数组的第i和i+1之和
"""
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows==0:
            return []  # return [1]
        tem = [0,1] # 初始化临时数组第0层，不加入到res中
        res = []
        for i in range(numRows): # 层数  # for i in range(numRows+1):
            rowlist = []
            for j in range(len(tem)-1): # 每一层上的元素个数=上一层个数+1=临时层个数-1
                rowlist.append(tem[j]+tem[j+1])
            res.append(rowlist)
            tem = rowlist[:]  # 一定得是深复制，作为下次循环的上一层
            tem.insert(0,0)  # 首端加上0
            tem.append(0)    # 尾端加上0
        return res  # return res[numRows]

119. Pascal's Triangle II
"""
Input: 3
Output: [1,3,3,1]
返回上一题创建矩阵中的索引为3的行,见上题#
"""

162. Find Peak Element 寻找山峰 easy
"""
Input: nums = [1,2,1,3,5,6,4]
Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6..
"""
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums or len(nums)==1:
            return 0
        i = 1
        while i < len(nums)-1:
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                return i
            else:
                i += 1
        if i == len(nums)-1 and nums[i] > nums[i-1]:
            return i
        return 0

163. Missing Ranges  缺失区间
"""
Input: nums = 
[0, 1, 3, 50, 75]
, lower = 0 and upper = 99,
Output: 
["2", "4->49", "51->74", "76->99"]
当区间长度为1和长度>2时，区间的格式不同，用 get_range函数来返回不同的格式
if cur - pre >=2:  # 说明cur与pre之间有空缺
    res.append( getRange(pre+1,cur-1))
"""
class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: List[str]
        """
        def get_range(start,end):   # return "{}->{}".format(x,y)
            if start == end:
                return "{}".format(start)
            else:
                return "{}->{}".format(start,end)
        res = []
        pre = lower-1 # 假设lower为2，nums[0]为3，那么 2也属于空缺区间。因此这里要-1
        for i in xrange(len(nums)):
            cur = nums[i]
            if cur - pre >=2:  # 说明cur与pre之间有空缺
                res.append( getRange(pre+1,cur-1))
            pre = cur
        if (upper+1) - nums[-1] >= 2:
            res.append(getRange(nums[-1] + 1, upper))


219. Contains Duplicate II   是否存在 不同下标的两个元素值相同 （dict：元素值作为 key，元素下标作为 value）
"""
给定一个数组，和一个整数k，判断该数组中是否存在不同下标的 i 和 j 两个元素，使得 nums[i] = nums[j]，且 i 和 j 的差不超过k。
遍历所有元素，将元素值当做键、元素下标当做值，存放在一个字典中。遍历的时候，如果发现重复元素，则比较其下标的差值是否小于k，
如果小于则可直接返回True，否则更新字典中该键的值为新的下标。
"""
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        num_map = {}
        for i in xrange(len(nums)):
            if nums[i] in num_map and i - num_map[nums[i]] <= k:
                return True
            else:
                num_map[nums[i]] = i
        return False

209. Minimum Size Subarray Sum 最短子数组之和：找出一个数组中最短连续的子数组，这个子数组的和要>=s.
"""
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
找不到则返回0
"""
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        l,r == 0,0
        csum = 0
        res = float('inf')
        while r < len(nums):
            csum += nums[r]
            while csum >= s:
                res = min(res,r-l+1) # res始终保存此时得到的最短距离
                csum -= nums[l] # 去掉此时左边的第一个，并将l+1，相当于l向右移动了一位；不断重复此过程，直到该区间之和<s
                l += 1          # 不会出现l>r的情况，因此在这之前必然有 csum < s
            r += 1
        if res != float('inf'):
            return res
        else:
            return 0

243.Shortest Word Distance  单词之间的最短距离
"""
Assume that words = ["practice", "makes", "perfect", "coding", "makes"]. 
Given word1 = “coding”, word2 = “practice”, return 3. 
Given word1 = "makes", word2 = "coding", return 1. 
单词可能重复出现
思路：记录单词1,2每次出现的位置，并记录
"""

class Solution:
    # @param {string[]} words
    # @param {string} word1
    # @param {string} word2
    # @return {integer}
    def shortestDistance(self, words, word1, word2):
        dist = float("inf")
        i, index1, index2 = 0, None, None
        while i < len(words):
            if words[i] == word1:
                index1 = i
            elif words[i] == word2:
                index2 = i
            if index1 is not None and index2 is not None:
                dist = min(dist, abs(index1 - index2)) # 后面的每次迭代都要进行计算，但这是不可避免的
            i += 1
        return dist

624.Maximum Distance in Arrays

"""
给m个已经排序好的数组，从两个数组中分别挑选一个数字，使得差最大，求最大的差 
Example 1: 
Input: 
[[1,2,3], [4,5], [1,2,3],[2,3,4]] 
Output: 4 
Explanation: 
从[1,2,3]中取1，[4,5]中取5。最大差为5-1=4
思路：
diff = max( diff, maxv-array[i][0],array[i][-1]-minv) 本组最大值-此时最小值，此时最大值-本组最小值
最小值来源于每个数组中下标为0的元素，最大值来源于每个数组中下标为-1的元素
"""
class Solution:
    def MaximumDistance(self,array,m):
        # minv和maxv用于记录已经遍历的list中的最小值与最大值，允许在同一个list中，比较不在同一list中进行
        minv = array[0][0]
        maxv = array[0][-1]
        diff = -1
        for i in range(1,m): # 从第二个数组开始遍历
            diff = max( diff, array[i][-1]-minv, maxv-array[i][0]) # 本组最大值-此时最小值，此时最大值-本组最小值
            maxv = max( maxv, array[i][-1]) # 更新此时的最大值与最小值
            minv = min( minv, array[i][0])
        return diff

4. 寻找两个有序数组的中位数
给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 nums1 和 nums2 不会同时为空。
示例 1:
nums1 = [1, 3]
nums2 = [2]
则中位数是 2.0

def median(A, B):
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) / 2
    while imin <= imax:
        i = (imin + imax) / 2
        j = half_len - i
        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0

作者：LeetCode
链接：https://leetcode-cn.com/problems/two-sum/solution/xun-zhao-liang-ge-you-xu-shu-zu-de-zhong-wei-shu-b/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

41. 缺失的第一个正数
给定一个未排序的整数数组，找出其中没有出现的最小的正整数。

示例 1:

输入: [1,2,0]
输出: 3
示例 2:

输入: [3,4,-1,1]
输出: 2

class Solution:
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if 1 not in nums:
            return 1
        # nums = [1]
        if n == 1:
            return 2
        # 用 1 替换负数，0，和大于 n 的数，在转换以后，nums 只会包含正数
        for i in range(n):
            if nums[i] <= 0 or nums[i] > n:
                nums[i] = 1
        # 使用索引和数字符号作为检查器
        # 例如，如果 nums[1] 是负数表示在数组中出现了数字 `1`
        # 如果 nums[2] 是正数 表示数字 2 没有出现
        for i in range(n):
            a = abs(nums[i])
            # 如果发现了一个数字 a - 改变第 a 个元素的符号
            # 注意重复元素只需操作一次
            if a == n:
                nums[0] = - abs(nums[0])
            else:
                nums[a] = - abs(nums[a])

        # 现在第一个正数的下标
        # 就是第一个缺失的数
        for i in range(1, n):
            if nums[i] > 0:
                return i

        if nums[0] > 0:
            return n

        return n + 1

作者：LeetCode
链接：https: // leetcode - cn.com / problems / two - sum / solution / que - shi - de - di - yi - ge - zheng - shu - by - leetcode /
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
1. Two Sum == target

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


11. Container With Most Water
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

15. 3Sum == 0

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
                    while (nums[left] == nums[left - 1] and nums[right] == nums[right + 1] and left < right):
                        left += 1
                elif sum > 0:
                    right -= 1
                    while (nums[right] == nums[right + 1] and left < right):
                        right -= 1
                else:
                    left += 1
                    while (nums[left] == nums[left - 1] and left < right):
                        left += 1
        return list(set(result))  # 去重

16. 3Sum Closest==target


class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums) < 3:
            return
        nums.sort()  # 先排序
        result = nums[0] + nums[1] + nums[2]
        for i in range(0, len(nums)):
            left = i + 1  #
            right = len(nums) - 1
            # 注意跳过与上轮重复的数！后面也有类似的处理
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum == target:
                    return sum
                elif sum > target:
                    if abs(sum-target) < abs(result-target):
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

18. 4Sum == target


class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        if len(nums) < 4:
            return result
        nums.sort()  # 先排序
        for k in range(len(nums)):
            if k > 0 and nums[k] == nums[k - 1]:
                continue
            for i in range(k+1, len(nums)):
                if i > k+1 and nums[i] == nums[i - 1]:
                    continue
                left = i + 1  #
                right = len(nums) - 1
                # 注意跳过重复的数！后面也有类似的处理
                while left < right:
                    sum =  nums[k] + nums[i] + nums[left] + nums[right]
                    if sum == target:
                        result.append((nums[k],nums[i], nums[left], nums[right]))
                        # 注意：此时应该同时移动left和right，因为此时绝对不可能出现left和right前一个匹配的情况
                        # 匹配上那也是重复的一对组合
                        left += 1
                        right -= 1
                        # 若出现大量的重复情况，改变left和right的一个就好了
                        while nums[left] == nums[left - 1]  and left < right:
                            left += 1
                        while nums[right] == nums[right + 1] and left < right:
                            right -= 1
                    elif sum > target:
                        right -= 1
                    else:
                        left += 1
        return result


26. Remove Duplicates from Sorted Array 排序列表去重

# 返回去重之后的list
# -*- coding:utf-8 -*-
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        result = []
        while i < len(nums)-1:
            while i < len(nums)-1 and nums[i] == nums[i+1]:
                i += 1
            result.append(nums[i])  # count += 1
            i += 1
        # 考虑最后一个元素
        if i>0  and i<len(nums) and nums[i] != nums[i-1]:
            result.append(nums[i]) # count += 1
        return result
c = Solution()
print c.removeDuplicates([1,1,2])

# 返回去重之后的列表长度以及列表：实现空间复杂度为1的操作
# 将每个第一次出现的元素移到j所在的位置的后一位，同时将j加1

class Solution(object):
    def removeDuplicates(self, nums):
        if len(nums)==0:
            return 0
        j=0
        for i in range(1,len(nums)):
            if nums[i]!=nums[j]:
                nums[j+1]=nums[i]
                j = j+1
        return  j+1,nums[:j+1]
c = Solution()
print c.removeDuplicates([1,1,2,4,5,6,6,6,7,7])


27. Remove Element 删除给定值val并返回数组长度
# 只要你不是val，那么我就把你放到j的位置，同时j向后移动一步

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
        for i in range(0,len(nums)):
            if nums[i] != val:
                nums[j] = nums[i]
                j = j+1
        return j

31. Next Permutation
"""
输出字典序中的下一个排列。比如123生成的全排列是：123，132，213，231，312，321。那么321的next permutation是123。
"""
# -*- coding:utf-8 -*-
# https://blog.csdn.net/qq_28119401/article/details/52972616
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

# 下面的写法在我自己的ide上是完美通过的。。。但是leetcode没过

class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        temp = map(str,nums) # 得到一个包含所有字符的list
        temp = ''.join(temp)
        L = list() # 用来存放最后的字符串
        path = ''  # path是已排好的组合
        self.connect(temp,L,path) # ss是剩下的待排序的字符

        uniq = list(set(L))  # 去重并转为整型
        uniq.sort()
        for i in range(len(uniq)):
            if uniq[i] == temp and i+1<len(uniq):
                temp1 = map(int,[j for j in uniq[i+1]])
                return temp1
            else:
                temp1 = map(int, [j for j in uniq[0]])
                return temp1

    def connect(self,ss,L,path):
        if ss=='':
            L.append(path)
        else:
            for i in range(len(ss)): # 每次拿出一个字符，然后将剩下字符的组合+此时拿出的字符
                self.connect(ss[:i]+ss[i+1:],L,path+ss[i])
c = Solution()
print c.nextPermutation([1,2,3])


33. Search in Rotated Sorted Array（试一试81题的方法！！！）
"""
在旋转数组中（无重复值）寻找某值
"""

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        start = 0
        end = len(nums) - 1
        while start <= end:
            mid = (start + end) / 2
            if nums[mid] == target:
                return mid
            # 判断此时取到的num[mid]属于右边升序序列还是左边升序序列
            if nums[mid] < nums[end]:  # nums[mid]属于右边升序序列
                # 判断target在num[mid]的右侧还是左侧
                if target > nums[mid] and target <= nums[end]: # target在num[mid]的右侧
                    start = mid + 1
                else:
                    end = mid - 1
            if nums[mid] >= nums[start]:  # nums[mid]属于左边升序序列
                if target >= nums[start] and target < nums[mid]: # # target在num[mid]的左侧
                    end = mid - 1
                else:
                    start = mid + 1
        return -1

34. Find First and Last Position of Element in Sorted Array

class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        first = self.get_first(nums,target,0,len(nums)-1)
        last = self.get_last(nums,target,0,len(nums)-1)
        return [first,last]

    def get_first(self,data,k,start,end):
        if start>end: # 没找到，返回-1
            return -1
        mid = (start+end)/2
        findk = data[mid]
        if findk == k:
            if mid>0 and data[mid-1]!=k or mid==0:
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

35. Search Insert Position

"""
找到某个数字应该放在此有序数组中的位置
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
        while start<=end:
            mid = (start+end)/2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid-1
            else:
                start = mid + 1
        return start

39. Combination Sum
"""
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
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
        self.dfs(candidates,[],target,0)
        return self.res

    def dfs(self,candidates,sub,target,last): # sub:本组，target：减去已选中数字，last:上轮选中的数字
                                              # （下次选的数字要>last，否则会造成多组重复）
        if target == 0:  # 目标变为0，说明已凑好一组
            self.res.append(sub[:])  # sub[:]==sub！！！！！！！！但是换成sub就全部变成空的了！！！！！！！！！！！！！百思不得其解啊！
        if target < candidates[0]: # 目标不为0且小于此时最小值，说明此组无法凑成
            return
        for n in candidates:
            if n > target: # 后面的数字肯定也不行
                return
            if n < last:
                continue
            sub.append(n)
            self.dfs(candidates,sub,target-n,n)
            # 注意此处不可采取candidates[:n]+[n+1:]的形式，因为要保证可以重复选取一个元素
            # 加入上面返回的元素失败了或成功了，弹出这一项，继续探索，进行下一个循环
            sub.pop()

40. Combination Sum II
"""
在39的基础上，增加一项条件，每个数字最多只能使用一次。如果本来就有两个数字
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
        self.res = [] # 相当于定义一个全局变量，可以在类中所有函数中使用
        candidates = sorted(candidates) # 从小的开始选
        used = [0]*len(candidates) # 用过的数字就标记为1
        self.dfs(candidates,[],target,used,0)
        return self.res

    def dfs(self,candidates,sub,target,used,last): # sub:本组，target：减去已选中数字，last:上轮选中的数字
                                              # （下次选的数字要>last，否则会造成多组重复）
        if target == 0:  # 目标变为0，说明已凑好一组
            self.res.append(sub[:])
        if target < candidates[0]: # 目标不为0且小于此时最小值，说明此组无法凑成
            return
        l = None # 为了防止重复的比如两个1，那在一层递归只处理一次
        # unhashable type: 'list',list内部为list没办法通过set来去重
        for m in range(len(candidates)):
            n = candidates[m]
            if n > target:
              return
            if n < last or used[m] == 1 or l == n:
              # 第一轮m循环我选出来1 1 6，第二轮开始的时候，l=n了，我无法回去选到之前的那个1，这样就不会出现重复选出1 1 6的情况了
              # 因为第二轮开始的时候 used[m]已经全变为0了此时为了防止重复只能用l来防止
              continue
            sub.append(n)
            used[m] = 1  # 记录是否 用过的
            self.dfs(candidates, sub, target - n, used, n)
            l = n
            used[m] = 0
            sub.pop()

48. Rotate Image
"""
把正方形矩阵原地顺时针旋转90度。
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
        # 做上下翻转
        for i in xrange(rows/2):
            for j in xrange(cols): # 第i行对应的行为rows-1-i，rows为整数，rows/2得到的也是整数
                matrix[i][j],matrix[rows-1-i][j] = matrix[rows-1-i][j],matrix[i][j]
        # 做对角线翻转
        for i in xrange(rows):
            for j in xrange(i):
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]

53. Maximum Subarray
"""
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
给你一串数，有正有负 要求出最大的连续子串。
可以当作是一个简单的动态规划问题：
局部最大值temp_max代表以当前位置为结尾的最大连续子串的最大值（必须包含当前结尾位置:这样才能保证是一个连续序列）
此时的局部最大值 = max(之前的局部最大值+结尾位置值 ，结尾位置值)
最大值 = max(temp_max，之前的max)
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

54. Spiral Matrix（顺时针打印矩阵，做法比剑指上好多了！！！）
"""
Input:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
输出螺旋矩阵的值。。。作业帮笔试的原题...错过了
原理：一层一层向里处理，依次是，最上层、最右层、最下层、最左层，即这一圈处理完毕，循环进入下一个内圈，直到结束
"""


class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if matrix == []:
            return []
        top = 0
        bottom = len(matrix)-1
        left = 0
        right = len(matrix[0])-1
        res = []
        while top<bottom and left<right:
            res += matrix[top][left:right+1]  # 输出最上面一行数据（学到了，居然还有这种方法）
            for x in range(top+1,bottom+1):     # 输出最右边一列数据
                res.append(matrix[x][right])
            res += matrix[bottom][left:right][::-1]  # 输出最下面一行数据
            for x in range(bottom-1,top,-1):   # 输出最左边一列数据
                res.append(matrix[x][left])
            top,bottom,left,right = top+1,bottom-1,left+1,right-1
        if top == bottom: # 若最后只剩下最后一行数据
            res += matrix[top][left:right+1]
        elif left == right: # 若只剩下一列数据
            for x in range(top,bottom+1):
                res.append(matrix[x][left])
        return res

55. Jump Game
"""
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
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
            if nums[i] >= tmp-i: #若从倒数第二个能走到最后一个，那么终点位置定为倒数第二个，直到找到一个能走到终点的最近的位置(贪心法)
                tmp = i          # 再将此位置定为新的终点，继续向前寻找一个能走到终点的最近的位置
        if tmp == 0:             # 若循环结束之后，终点位置为起点，则说明能到达
            return True
        return False

56. Merge Intervals
"""
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
"""

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if not intervals:
            return []
        if len(intervals) < 2:
            return intervals  # [][0]会报错
        # 首先按照起点位置排序
        intervals = sorted(intervals,key=lambda x:x[0])
        # 排好序之后判断一个区间的起点是否在前一个区间中，若在则合并前一个区间与此时区间，若不在，则将新区间添加。
        res = [intervals[0]]
        pre = res[0]
        for elem in intervals[1:]:
            if pre[1] >= elem[0]:
                if pre[1] <= elem[1]:
                    pre[1] = elem[1] # 将此时res中的最后一项（pre区间）的终点设为elem的终点（相当于合并两个区间并替换此时的pre）
                                     # 若 pre[1]>elem[1]，没必要更新res最后一项的终点
            else: # 毫无交集的情况，将此时的区间加到res中
                res.append(elem)
                pre = res[-1]  # 此时pre取得res中的最后一项，注意，和初始化时采用pre=res[0]一样，接下来对pre的修改会直接修改res中的值
        return res

59. Spiral Matrix II
"""
Given a positive integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
Input: 3
Output:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
"""
class Solution(object):
    def generateMatrix(self, n):
        if not n:
            return []
        res = [[0 for _ in xrange(n)] for _ in xrange(n)]
        left, right, top, down, num = 0, n-1, 0, n-1, 1
        while left <= right and top <= down:
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

# 类似于剑指上的实现（能跑但太复杂了，还是leetcode上大佬多！！！）
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if n==0:
            return []
        m = n
        matrix = [ [0 for x in range(n)] for y in range(m)] # 延伸到m行n列的矩阵
        index = 1 # 要填充的数字
        round = (n+1)/2  # round：圈数，3*3的矩阵，圈数为2
        for x in range(0,round): # x代表此时圈出发的横坐标，同时也是纵坐标，（x,x）
            for y in range(x,n-x):           # ——>部分
                matrix[x][y] = index                       # 横坐标（左面）发生变化
                index += 1
            for y in range(x+1,m-x-1):       #  |
                matrix[y][n-x-1] = index     #  |          # 纵坐标发生（上面）变化
                index += 1                   # \_/ 部分
            if n-2*x>1:  # 存在 <---的部分，这个规则和剑指中差不多，总结出来的（别人写的是m-2*x>1）
                for y in range(n-x-1,x-1,-1):  # n-x-1，因为右侧有一个已经填了
                    matrix[m-x-1][y] = index               # 横坐标发生变化
                    index += 1
            if m-2*x>1:
                for y in range(m-x-2,x,-1):    # n-x-2，因为上下两个都填了
                    matrix[y][x] = index
                    index += 1
        return matrix

62. Unique Paths
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
        dp = [ [0 for col in range(n)] for row in range(m)] # dp = [[0] * (n + 1) for row in range(m + 1)]
        # 初始化边缘
        for i in range(n):
            dp[0][i] = 1   # 到达第一行的各个位置路径数均为1
        for i in range(m):
            dp[i][0] = 1
        for row in range(1,m):
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
        # 由于矩阵每一处都有可能是障碍，所以将边缘初始化为1行不通了，直接做循环（实际上是更全面的方法，加上一个判断是否为边界的即可）
        if obstacleGrid[0][0] == 1:
            return 0
        dp[0][0] = 1
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0   # 并直接进入下一次循环
                else:
                    if i != 0: # 判断是否为边界
                        dp[i][j] += dp[i-1][j]
                    if j != 0:
                        dp[i][j] += dp[i][j-1]
        return dp[m-1][n-1]

64. Minimum Path Sum
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
        dp = [ [0 for row in range(n)] for col in range(m)]
        for i in range(m):
            for j in range(n):
                temp1 = 100000
                temp2 = 100000
                if i==0 and j==0:
                    dp[i][j] = grid[0][0]
                    continue
                if i>0:
                    temp1 = dp[i-1][j]
                if j>0:
                    temp2 = dp[i][j-1]
                dp[i][j] = grid[i][j] + min(temp1,temp2)
        return dp[m-1][n-1]

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

73. Set Matrix Zeroes
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

74. Search a 2D Matrix
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
        for i in range(m):
            if target > matrix[i][n-1]:
                row = i + 1
            elif target == matrix[i][n-1]:
                return True
            else:
                break
        if row >= m:
            return False

        j = 0
        while j<n:
            if matrix[row][j] == target:
                return True
            elif matrix[row][j] < target:
                j += 1
            else:
                return False

75. Sort Colors
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

78. Subsets
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
                x = temp[:]  # 浅拷贝是x=temp，这里是深拷贝
                x.append(num)
                res.append(x)  # 第一轮大循环，res=[[],[1]]；第二轮x依次为[2],[1,2]
        return res

79. Word Search
"""
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
"""
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
            if word[index] == board[row][col]: # 将目前以找到的路径赋值为0
                board[row][col] = '#'
                # 若此时index为最后一个，说明所有的字符都找到了，返回True
                if index == len(word)-1 or \
                    dfs(index+1,row+1,col) or \
                    dfs(index+1,row-1,col) or \
                    dfs(index+1,row,col+1) or \
                    dfs(index+1,row,col-1) :
                    return True
                board[row][col] = word[index]
            return False

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]: # 找到起点
                    if dfs(0,i,j): # 0为index，是目前要找的字母，i，j为起点
                        return True
        return False

80. Remove Duplicates from Sorted Array II

"""
Given nums = [1,1,1,2,2,3],
Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
只允许两个重复的元素，多的去掉。
使用两个指针prev和curr，判断A[curr]是否和A[prev]、A[prev-1]相等，如果相等curr指针继续向后遍历，
直到不相等时，将curr指针指向的值赋值给A[prev+1]，这样多余的数就都被交换到后面去了。最后prev+1值就是数组的长度。

还有一个很鸡贼的方法：
类似于基数排序，统计各个元素个数，大于2将dict中的value改为2，小于2保留原value，累加起来就好。。。
这样还能输出最终得到的list（每个元素循环它对应的value次）
"""

class Solution(object):
    def removeDuplicates(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(A)<=2:
            return len(A)
        prev = 1 # 从第二个开始
        curr = 2 # 从第三个开始
        while curr < len(A):
            if A[curr] == A[prev] and A[curr] == A[prev-1]:
                curr += 1   # 此时返回prev+1，始终为2个数
            else:
                prev += 1
                A[prev] = A[curr]
                curr += 1
        return prev+1

81. Search in Rotated Sorted Array II
"""
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
含有重复数字的旋转数组中寻找某值.
思路：
在正式比较之前，先移动左指针，使他指向一个和右指针不同的数字上。然后再做33题的查找。

"""
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
                l += 1   # 使左指针指向一个和右指针不同的数字上
            mid = (l+r)/2
            if nums[mid] == target:
                return True
            if nums[mid] >= nums[l]:  # 说明从l到mid这一段是有序的
                if nums[l] <= target < nums[mid]: # 若目标在mid前面
                    r = mid - 1
                else:  # 目标在mid后面
                    l = mid + 1
            elif nums[mid] <= nums[r]: # 说明mid到r这一段是有序的
                if nums[mid] < target <= nums[r]:  # 目标在mid后面
                    l = mid + 1
                else:
                    r = mid - 1
        return False

88. Merge Sorted Array
"""
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
        while n1>=0 and n2>=0:
            if nums2[n2] >= nums1[n1]:
                nums1[n1+n2+1] = nums2[n2]
                n2 -= 1
            else:
                nums1[n1+n2+1] = nums1[n1]
                n1 -= 1
        if n1 < 0:
            nums1[:n2+1] = nums2[:n2+1]

90. Subsets II
"""
含有重复数字求其所有组合
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
                # 这个地方一定要带切片啊，因为如果不带分片，就是一个浅拷贝，后面res有变化，导致res长度不断增加，不能退出for循环
                x = temp[:]  # 这里是深拷贝,浅拷贝是x=temp
                x.append(num)
                if x not in res:
                    # 加了这一句，防止出现相同list,注意 [1,4,4,4]和[4,4,1,4]被当作不同list，因此前面要先对nums排序！
                    # 否则按照我们的生成规则，nums=[4,4,4,1]时会生成这两种，且无法去重
                    res.append(x)  # 第一轮大循环，res=[[],[1]]；第二轮x依次为[2],[1,2],res为[ [],[1],[2],[1,2] ]
        return res

118. Pascal's Triangle
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
设定一个临时数组，为上一层数组首尾各加0，rowlist第i个值为临时数组的第i和i+1之和
"""
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows==0:
            return []  # return [1]
        tem = [0,1]
        res = []
        for i in range(numRows): # 层数  # for i in range(numRows+1):
            rowlist = []
            for j in range(len(tem)-1): # 每一层上个数=上一层个数+1=临时层个数-1
                rowlist.append(tem[j]+tem[j+1])
            res.append(rowlist)
            tem = rowlist[:]
            tem.insert(0,0)  # 首端加上0
            tem.append(0)
        return res  # return res[numRows]

119. Pascal's Triangle II
"""
Input: 3
Output: [1,3,3,1]
返回上一题创建矩阵中的索引为3的行,见上题#
"""

120. Triangle
"""
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
左边的元素只能从上面上一行同一列下来，中间的可以从上一行前一列或同列，右边的只能从前一列
统计到达最后一行的所有路径长度，选出最短的即为最优路径
"""

class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        m = len(triangle)
        l = triangle   # l为新建的表格，存储矩阵中每一点的最优路径
        for i in range(1,m):  # 从索引为1的行（即第二行）开始
            for j in range(i+1): # 索引为i的行有i+1列
                if j == 0: # 最左边
                    l[i][j] = l[i-1][0] + triangle[i][0]  # 这个地方改为 l[i][0]，也可
                elif j == i: # 最右边 ,注意这里时elif！否则有可能执行两次
                    l[i][j] = l[i-1][i-1] + triangle[i][j]
                else:
                    l[i][j] = min(l[i-1][j-1],l[i-1][j]) + l[i][j]
        return min(l[m-1][:])+1

121. Best Time to Buy and Sell Stock
"""
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
"""
class Solution:
    class Solution(object):
        def maxProfit(self, prices):
            """
            :type prices: List[int]
            :rtype: int
            """
            if not prices or len(prices) == 1:
                return 0
            buy = prices[0]
            max_profit = prices[1]-prices[0]
            for i in range(2,len(prices)):
                buy = min(buy,prices[i-1])
                profit = prices[i]-buy
                max_profit = max(max_profit,profit)
            if max_profit<0:
                return 0
            return max_profit

122. Best Time to Buy and Sell Stock II
"""
Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
可以多次买入卖出。
寻找有序对的个数：7 1 4 5 3 6 2 3 5，有序对：145 36 235，最大收益 = 每个有序对首尾值之差
这样写太麻烦了。。。
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

123. Best Time to Buy and Sell Stock III
"""
Input: [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
             Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
You may complete at most two transactions.最多只有两次操作：即 买卖买卖，且买和卖不能在同一天。     
https://blog.csdn.net/u014251967/article/details/52517256       
"""

152. Maximum Product Subarray 连续子数组的最大乘积
"""
Input: [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
思路：
某个位置可能出现了0或者负数，遇到0时，整个乘积会变成0；遇到负数时，当前的最大乘积会变成最小乘积，最小乘积会变成最大乘积
使用两个数组分别记录以某个位置i结尾的时候的最大乘积和最小乘积了。令最大乘积为f，最小乘积为g。那么有：

当前的最大值等于已知的最大值与最小值和当前值的乘积、当前值，这三个数的最大值。
当前的最小值等于已知的最大值与最小值和当前值的乘积、当前值，这三个数的最小值。
结果是最大值数组中的最大值。
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
        f = [0]*n
        g = [0]*n
        f[0] = g[0] = res_max = nums[0]
        for i in range(1,n):
            f[i] = max(f[i - 1] * nums[i], nums[i], g[i - 1] * nums[i])
            g[i] = min(f[i - 1] * nums[i], nums[i], g[i - 1] * nums[i])
            res_max = max(f[i],res_max)
        return res_max

153. Find Minimum in Rotated Sorted Array  旋转数组中的最小值
"""
Input: [3,4,5,1,2] 
Output: 1
"""
class Solution(object):
    def findMin(self, rotateArray):
        # write code here
        l=len(rotateArray)
        if l==0:
            return 0
        if rotateArray[0]<rotateArray[l-1]:  # 若产生了旋转，必然>=。说明没有产生旋转，那么最前面的就是最小的元素
            return rotateArray[0]
        # 直接遍历吧（当第一个=最后一个=中间时，其他情况可以采用二分法），这种方法的速度为700ms（还好时间限制3秒）
        pre = -1e7
        for num in rotateArray:
            if num < pre:
                return num    # 由于两个子序列均递增，若出现一个比它小的，必为最小值
            pre = num
        return rotateArray[0]   # 如果大家都相等，就返回第一个值吧

154. Find Minimum in Rotated Sorted Array II
"""
Input: [2,2,2,0,1]
Output: 0
可能包含重复字符，与之前写法一致
"""

162. Find Peak Element
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

163. Missing Ranges（缺失区间）
"""
Input: nums = 
[0, 1, 3, 50, 75]
, lower = 0 and upper = 99,
Output: 
["2", "4->49", "51->74", "76->99"]
当区间长度为1和长度>2时，区间的格式不同，用 get_range函数来返回不同的格式
"""
class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: List[str]
        """
        def get_range(start,end):
            if start == end:
                return "{}".format(start)
            else:
                return "{}->{}".format(start,end)
        res = []
        pre = lower-1
        for i in xrange(len(nums)):
            cur = nums[i]
            if cur - pre >=2:
                res.append( getRange(pre+1,cur-1))
            pre = cur
        if (uppe+1) - nums[-1] >= 2:
            res.append(getRange(nums[-1] + 1, upper))

189. Rotate Array
"""
Input: [1,2,3,4,5,6,7] and k = 3
Output: [5,6,7,1,2,3,4]
"""
class Solution(object):
    def rotate(self, nums, k):
        n=len(nums)
        if n<2 or k==0:
            return
        k=k%n  # k为实际要放到前面去的元素个数
        nums[:]=nums[n-k:]+nums[:n-k]

219. Contains Duplicate II
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

209. Minimum Size Subarray Sum
"""

"""
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """

216. Combination Sum III
"""

"""


243.Shortest Word Distance
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
[[1,2,3], 
[4,5], 
[1,2,3]] 
Output: 4 
Explanation: 
从[1,2,3]中取1，[4,5]中取5。最大差为5-1=4
思路：
最小值来源于每个数组中下标为0的元素，最大值来源于每个数组中下标为-1的元素
"""
class Solution:
    def MaximumDistance(self,array,m):
        # minv和maxv用于记录已经遍历的list中的最小值与最大值，允许在同一个list中，比较不在同一list中进行
        minv = array[0][0]
        maxv = array[0][-1]
        diff = -1
        for i in range(1,m):
            diff = max( diff, maxv-array[i][0],array[i][-1]-minv) # 注意这里的比较始终在不同list中进行
            maxv = max( maxv, array[i][-1])
            minv = min( minv, array[i][0])
        return diff


import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 27、二叉树的镜像
"""
操作给定的二叉树，将其变换为源二叉树的镜像。
            8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
思路：不断交换两个子树，第一次交换后下面的分叉没有交换，递归调用并传入root.left和root.right,对每个子树的分叉再进行交换
       交换根结点的左右子树，交换6和10结点的左右子树....
"""

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点     ？？？ 不知道为啥，不能写return root。
    def Mirror(self, root):
        # write code here
        if root != None:
            root.left,root.right = root.right,root.left
            self.Mirror(root.left)
            self.Mirror(root.right)

# 28、对称的二叉树
"""
请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
"""

# 换种思路：直接对比每一棵树的左子树和右子树是否相同，递归。比较每一颗子树的根结点是否相同，
#  注意： 最后的时候，若二者同时变为None，则返回True。若不是同时返回None，返回False
#  让两课相同的树做这种比较
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        # write code here
        return self.isSym(pRoot,pRoot)

    def isSym(self,tree1,tree2):
        if tree1 is None and tree2 is None:
            return True
        if tree1 is None or tree2 is None:
            return False
        if tree1.val != tree2.val:
            return False
        return self.isSym(tree1.left,tree2.right) and self.isSym(tree1.right,tree2.left)

# 29、顺时针打印矩阵

"""
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字(见剑指offer P182)
"""
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        if matrix == None:
            return None
        rows = len(matrix)
        cols = len(matrix[0])
        start = 0 # 新的一圈的起始坐标 行列号总是相同的
        result = [] # 用于存放最终输出的数字
        while rows > 2*start and cols > 2*start:
            endx = rows-1-start
            endy = cols-1-start
            for i in range(start,endy+1):
                result.append(matrix[start][i])
            if start < endx:
                for i in range(start+1,endx+1):
                    result.append(matrix[i][endy])
            if start < endx and start < endy:
                for i in range(endy-1,start-1,-1):     # 输出的最后一个元素纵坐标为start，逆序，索引为start-1
                    result.append(matrix[endx][i])
            if start < endx-1 and start < endy:
                for i in range(endx-1,start,-1):       # 不输出到纵坐标为start的位置，坐标为start+1，故索引为start+1-1=start
                    result.append(matrix[i][start])
            start = start+1
        return result

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
    def push(self, node):
        # write code here
        self.stack.append(node)
        if not self.min_stack or node <= self.min_stack[-1]:  # 注意：写，if self.min_stack is None 报错了！能用not就用not
            self.min_stack.append(node)
    def pop(self):
        # write code here
        if self.stack[-1] == self.min_stack[-1]: # 若数据栈弹出的那一项碰巧是辅助栈栈顶（最小值），则弹出辅助栈栈顶
            self.min_stack.pop()   # 这个pop函数前面没有self.，表明调用的是基本库中的pop函数
        self.stack.pop()
    def top(self):
        # write code here
        # 不是弹出，只是显示栈顶
        return self.stack[-1]
    def min(self):
        # write code here
        return self.min_stack[-1]

# 31、栈的压入、弹出序列
"""
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是弹出序列。
（注意：这两个序列的长度是相等的，压入顺序不变，但可以压入部分元素后，再弹出某元素）
"""
"""
画图，思路更清晰：如果下一个弹出的数字刚好是栈顶数字，那么直接弹出；若不是，则先压入再弹出；
若始终没找到要弹出的，说明不可能是弹出序列。一定要用例子自己根据程序跑一遍
"""
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        if not pushV or len(pushV)!= len(popV):
            return False
        stack = []
        j = 0
        for i in range(len(pushV)):
            stack.append(pushV[i])
            while stack and stack[-1] == popV[j]: # 之前写的if语句，只能重复一次这个过程，不适应后面321的弹出
                j = j + 1
                stack.pop()
        # 等循环结束后再来判断stack是否为空，不能写在循环内部
        if not stack:
            return True

# 32、从上到下打印二叉树（广度优先搜索BFS）
"""
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
"""
"""
思路：利用队列，每次打印一个节点的时候，如果该节点有子节点，则将其左右子节点添加至队列末尾，比如打印左子节点，又把左子节点的
左右子节点添加至队列末尾。直到队列中所有节点被打印完。
"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return []    # 注意：结果要求的是返回一个list，因此这里必须返回一个空列表，否则这个case就通过不了
        l = list() # 用于按顺序存储节点
        q = [root] # 将头指针存储在队列中(用一个list来实现即可),就算后期insert一个位置还没产生的地方也可以
        while q:
            temp = q.pop(0) # 将q此时的结点保存到temp中，并将其从temp中弹出，很关键的一步，要不然q始终不为空
            l.append(temp.val)
            if temp.left:
                q.append(temp.left)
            if temp.right:
                q.append(temp.right)
        return l
"""
分行打印节点：在上个代码基础上增加两个变量即可。unprinted表示当前层中未打印的节点数，初值为1，每打印一个节点，它减1。
nextlevel表示下一行的节点数，初值为0，若当前节点有子节点，子节点加入队列的同时将nextlevel加1,。
unprinted变为0的时候，输出换行（将当前行的元素l添加到最终列表final中），将nextlevel赋值给unprinted。
"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, root):
        # write code here
        if not root:
            return []
        l = []
        q = [root]
        final = list()  # final用来存储每一层的list
        unprinted = 1
        nextlevel = 0
        while q:
            temp = q.pop(0)
            l.append(temp.val)
            unprinted -= 1
            if temp.left:
                q.append(temp.left)
                nextlevel += 1
            if temp.right:
                q.append(temp.right)
                nextlevel += 1
            if unprinted == 0:
                final.append(l)
                l=[]
                unprinted = nextlevel
                nextlevel = 0     # 注意：花个五分钟跑一个demo就能发现之前可能是一个非常简单的小bug
        return final
"""
请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，
第三行按照从左到右的顺序打印，其他行以此类推。在分层输出的代码中增加一个变量i即可。根据i的奇偶来确定是否reverse。
"""
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, root):
        # write code here
        if not root:
            return []
        l = []
        q = [root]
        final = list()
        unprinted = 1 # unprinted表示当前层中未打印的节点数
        nextlevel = 0 # nextlevel表示下一行的节点数
        i = 1 # i为奇数，则list不reverse，i为偶数，则reverse
        while q:
            temp = q.pop(0)
            l.append(temp.val)
            unprinted -= 1
            if temp.left:
                q.append(temp.left)
                nextlevel += 1
            if temp.right:
                q.append(temp.right)
                nextlevel += 1
            if unprinted == 0:
                if i % 2 == 0:
                    l.reverse()
                final.append(l)
                l=[]
                unprinted = nextlevel
                nextlevel = 0
                i += 1
        return final

# 33、二叉搜索树的后序遍历序列
"""
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
二叉搜索树的特点：左子树节点的值<=根节点的值，右子树节点的值>=根节点的值,注意这里符合<>关系
"""
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        length = len(sequence)
        if length == 0:
            return False
        if length == 1:
            return True
        root = sequence[-1]  # 后序遍历的最后一个节点为根节点
        left = 0
        while sequence[left] < root:
            left += 1
        # 此时left取得右子树的第一个坐标
        for j in range(left,length-1): # 对右子树做循环
            if sequence[j]<root:
                return False
        # 对每一棵子树进行递归操作
        return self.VerifySquenceOfBST(sequence[:left]) or self.VerifySquenceOfBST(sequence[left:length-1])
        # 为啥这里要用or？？？？？？不应该用and吗？？？


# 34、二叉树中和为某一值的路径
"""
输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和 为 输入的整数的所有路径。
路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
"""
"""
思路：不断递归，在子树中查找 符合 expectNumber-父结点.val 的值
"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        # 树为空，返回空列表
        if not root:
            return []
        # 单点树且符合，返回根节点值
        if root and not root.left and not root.right and root.val==expectNumber:
            return [[root.val]]
        res = []
        left = self.FindPath(root.left,expectNumber-root.val) # 先在左子树里找
        right = self.FindPath(root.right,expectNumber-root.val) # 也在右子树里找
        for i in left+right: # i符合[xxx]形式
            res.append([root.val]+i)   # [5]+[6]就是[5,6],这样好像并不能保证数组长度大的靠前，不过能通过
        return res

# 35、复杂链表的复制（分解让复杂问题简单化）
"""
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
返回结果为复制后复杂链表的head。（注意！输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

# O（n）的空间，O（n）的时间：复制原始链表上的每个节点，将<N,N`>放在一个哈希表中，N指向S，N`就指向S~
# O（1）的空间：每个复制出来的N~放在N后面
"""
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if not pHead:
            return  pHead
        # 对每一个节点进行逐一复制
        p = RandomListNode(pHead.label) # 定义一个头指针与pHead相同的RandomListNode类
        p.next = pHead.next      # 复制这个节点的next指针
        p.random = pHead.random  # 复制这个节点的random指针

        p.next = self.Clone(pHead.next) # 对下一个节点进行复制
        return p

# 36、二叉搜索树与双向链表（没看完，有点复杂）
"""
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
"""
"""
二叉树中每个节点都有两个指向子节点的指针，双向链表中每个节点的两个指针分别指向前一个节点和后一个节点。能实现二者的转换
二叉搜索树中，左子节点的值<父节点的值，右子节点的值>父节点的值，
因此转换为双向链表时，原本指向左子节点的指针调整为链表中指向前一个节点的指针(按从小到大顺序排嘛)。
中序遍历每个节点，得到的便是从小到大排好序的。然后将他们用left和right指针相互连接起来就好了。
"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def NodeList(self, pRootOfTree):
        # write code here
        if not pRootOfTree:
            return []
        return self.NodeList(pRootOfTree.left) + [pRootOfTree] + self.NodeList(pRootOfTree.right)

    def Convert(self,pRootOfTree):
        # write code here
        res = self.NodeList(pRootOfTree) # 获取中序遍历
        if len(res) == 0:
            return None
        if len(res) == 1:
            return pRootOfTree
        # 对第一个节点和最后一个节点操作
        res[0].left = None
        res[0].right = res[1]
        res[-1].left = res[-2]
        res[-1].right = None
        # 对剩下的结点进行操作
        for i in range(1, len(res) - 1):
            res[i].left = res[i - 1]
            res[i].right = res[i + 1]
        return res[0]


# 37、序列化二叉树
"""
请实现两个函数，分别用来序列化和反序列化二叉树
"""
"""
所谓序列化，就是用一组序列（如字符串）来表示一个二叉树:叶子节点的两个左右子节点就是两个#，单枝就是跟一个#，根节点与子节点之间用，连接
反序列化将此序列恢复为二叉树
"""

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Serialize(self, root):
        # write code here
        if not root: # 左右子节点为None，用#代替
            return '#'
        return str(root.val) + ',' + self.Serialize(root.left) + self.Serialize(root.right) # 中序遍历

    # 反序列化，没看懂。。。
    flag = -1
    def Deserialize(self, s):
        # write code here
        self.flag += 1

        l = s.split(',')
        if self.flag >= len(s):
            return None

        root = None
        if l[self.flag] != '#':
            root = TreeNode(int(l[self.flag]))
            root.left = self.Deserialize(s)
            root.right = self.Deserialize(s)
        return root  # 如果是#,则返回None
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
        L = list() # 用来存放最后的字符串
        self.connect(ss,L,'')

        uniq = list(set(L))  # 由于可能会有重复的字符，也就有重复的字符串，需要转为set，再转为list
        return sorted(uniq)    # 并按序排列

    def connect(self,ss,L,path):
        # 每次拿出一个字符，然后将剩下字符的组合+此时拿出的字符
        # 最后一次，ss[i]取到了最后一个字符，此时调用的函数ss变为''，说明递归到了最底层，
        # 此时path=之前所有字符的叠加，append到L
        if ss=='':
            L.append(path)
        else:
            for i in range(len(ss)):
                self.connect(ss[:i]+ss[i+1:],L,path+ss[i])

"""
举一反三！！！
若面试题是按照一定的要求摆放若干个数字，则可以求出这些数字的所有排列，然后一一判断每个排列是不是满足题目给定的要求
例如：P219，正方体顶点放置数字问题，8皇后问题！！！
"""
"""
若改为求字符的所有组合（不考虑顺序，长度为1-n），abc的组合有：a,b,c,ab,ac,bc,abc.ab和ba属于同一组合。
思路：输入n个字符，则能构成长度为m的组合，1<=m<=n，我们将这n个字符分成两部分：第一个字符和其余的所有字符。
若组合中包含第一个字符，则下一步在剩余的字符里选择m-1个字符，若组合里不包括第一个字符，则下一步在剩余的字符里选m个字符
这样，就分解为了两个子问题：n中求m个字符的组合，n中求m-1个字符的组合
# Python实现根据 https://blog.csdn.net/MDreamlove/article/details/79528836 改编而来
"""

# -*- coding:utf-8 -*-
class Solution:
    def Combination(self, ss):
        # write code here
        if len(ss) <= 0:
            return []
        global final
        final = list() # 用来存放最后的字符串
        L=list()  # 用来存放长度为num个字符时的组合
        for j in range(1,len(ss)+1):
            self.connect(ss, 0, j, L) # 从字符串中第一个字符开始，依次取num个，1 <= num <=len(ss)
        return final

    def connect(self, str, begin, num, L):
        if str==None or len(str) == 0:
            return
        if num==0:  # num为0，说明已经凑够了num个字符，直接输出并返回
            final.extend(L)
        if begin > len(str)-1:
            return
        L.append(str[begin])   # 选中当前字符
        self.connect(str,begin+1,num-1,L)
        del L[len(str)-1]  # 当前（位置）字符未被选中，从最后一个位置开始，删去,然后查找其他组合
        self.connect(str, begin + 1, num, L)

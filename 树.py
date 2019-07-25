# -*- coding:utf-8 -*-
# 定义树结构
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 前序遍历

def pre_iter(node):
    if not node:
        return
    print(node.val)  # list1.append(node.val)
    pre_iter(node.left)
    pre_iter(node.right)

# 先序打印二叉树（非递归）
def preOrderTravese(node):
    stack = [node]
    while len(stack) > 0:
        print(node.val)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
        node = stack.pop()

# 中序遍历

def mid_iter(node):
    if not node:
        return
    mid_iter(node.left)
    print(node.val)
    mid_iter(node.right)

# 中序打印二叉树（非递归）
def inOrderTraverse(node):
    stack = []
    pos = node
    while pos is not None or len(stack) > 0:
        if pos is not None:
            stack.append(pos)
            pos = pos.left
        else:
            pos = stack.pop()
            print(pos.val)
            pos = pos.right

# 另一种方法：这样得到的是保留了树的结构的中序遍历
def NodeList(pRootOfTree):
    # write code here
    if not pRootOfTree:
        return []
    return NodeList(pRootOfTree.left) + [pRootOfTree] + NodeList(pRootOfTree.right)

# 后序遍历

def post_iter(node):
    if not node:
        return
    post_iter(node.left)
    post_iter(node.right)
    print(node.val)

# 后序打印二叉树（非递归）
# 使用两个栈结构
# 第一个栈进栈顺序：左节点->右节点->根节点
# 第一个栈弹出顺序： 根节点->右节点->左节点(先序遍历栈弹出顺序：根->左->右)
# 第二个栈存储为第一个栈的每个弹出依次进栈
# 最后第二个栈依次出栈
def postOrderTraverse(node):
    stack = [node]
    stack2 = []
    while len(stack) > 0:
        node = stack.pop()
        stack2.append(node)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    while len(stack2) > 0:
        print(stack2.pop().val)

# 按层遍历
# 先进先出选用队列结构
import queue
def layerTraverse(head):
    if not head:
        return None
    que = queue.Queue()      # 创建先进先出队列
    que.put(head)
    while not que.empty():
        head = que.get()    # 弹出第一个元素并打印
        print(head.val)
        if head.left:       # 若该节点存在左子节点,则加入队列（先push左节点）
            que.put(head.left)
        if head.right:      # 若该节点存在右子节点,则加入队列（再push右节点）
            que.put(head.right)

# 求二叉树节点个数
def treeNodenums(node):
    if node is None:
        return 0
    nums = treeNodenums(node.left)
    nums += treeNodenums(node.right)
    return nums + 1

# 二叉树的最大深度
def bTreeDepth(node):
    if node is None:
        return 0
    ldepth = bTreeDepth(node.left)
    rdepth = bTreeDepth(node.right)
    return (max(ldepth, rdepth) + 1)

# 7、重建二叉树

""""
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
"""


class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin): # pre和tin为list
        # write code here
        if len(pre) == 0: # 空树
            return None
        if len(pre) == 1: # 单点树（深度为0），已经到了最后一层，返回此时的叶节点
            return TreeNode(pre[0])
        else:
            # 先序遍历重建代码：利用reConstructBinaryTree(前，中)
            val = pre[0]  # 取得根结点的值
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

# 26、树的子结构（判断B是不是A的某一部分）
"""
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
return self.is_subtree(pRoot1,pRoot2) or self.HasSubtree(pRoot1.left,pRoot2) or self.HasSubtree(pRoot1.right,pRoot2)
"""

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if not pRoot1 or not pRoot2: # 如果二者有任意一个是空的
            return False
        return self.is_subtree(pRoot1,pRoot2) or self.HasSubtree(pRoot1.left,pRoot2) or self.HasSubtree(pRoot1.right,pRoot2)
            # 注意后面两个递归调用的是 HasSubtree函数！

    def is_subtree(self,A,B):
        # 用来判断B是否为A的一部分
        # 里面包含着self.is_subtree(A.left,B.left) and self.is_subtree(A.right,B.right) 的递归
        if B is None:
            return True
        if A is None or A.val != B.val: # B不为空而A为空 或 此两个节点值不同
            return False
        # 在树A中找到了和树B的根结点的值一样的结点R，继续比较二者的左右子树是否完全相同
        return self.is_subtree(A.left,B.left) and self.is_subtree(A.right,B.right)

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
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if root:
            root.left,root.right = root.right,root.left
            self.Mirror(root.left)
            self.Mirror(root.right)

# 28、对称的二叉树
"""
请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
"""

# 换种思路：直接对比每一棵树的左子树和右子树是否相同，递归比较每一颗子树的根结点是否相同，
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
        return self.isSym(pRoot,pRoot) # 复制一下

    def isSym(self,tree1,tree2):
        if tree1 is None and tree2 is None:
            return True
        if tree1 is None or tree2 is None: # 没有同时为空
            return False
        if tree1.val != tree2.val:
            return False
        return self.isSym(tree1.left,tree2.right) and self.isSym(tree1.right,tree2.left)

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
        q = [root] # 将头指针存储在队列中(用一个list来实现即可),右边append入，左边pop出
        while q:
            temp = q.pop(0) # 将q此时的结点保存到temp中，并将其从temp中弹出，很关键的一步，要不然q始终不为空
            l.append(temp.val)
            if temp.left:
                q.append(temp.left)
            if temp.right:
                q.append(temp.right)
        return l

# 二叉树的深度优先遍历(DFS)

class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def dfs(self, root):
        # write code here
        if not root:
            return []    # 注意：结果要求的是返回一个list，因此这里必须返回一个空列表，否则这个case就通过不了
        l = list() # 用于按顺序存储节点
        q = [root] # 将头指针存储在栈中，用list实现即可
        while q:
            temp = q.pop() # 栈是先入后出，右边append入，右边pop出
            l.append(temp.val)
            if temp.right:  # 右子节点先入栈（后出来），意味着先深度遍历左子树
                q.append(temp.right)
            if temp.left:
                q.append(temp.left)
        return l

"""
分行打印节点：在上个代码基础上增加两个变量即可。unprinted表示当前层中未打印的节点数，初值为1，每打印一个节点，它减1。
nextlevel表示下一行的节点数，初值为0，若当前节点有子节点，子节点加入队列的同时将nextlevel加1,。
unprinted变为0的时候，输出换行（将当前行的元素l添加到最终列表final中），将nextlevel赋值给unprinted,nextlevel=[]。
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

# 33、判断某数组是否为二叉搜索树的后序遍历序列
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
        return self.VerifySquenceOfBST1(sequence)

    def VerifySquenceOfBST1(self, sequence):
        # write code here
        length = len(sequence)
        if length == 0 or length == 1:
            return True
        root = sequence[-1]  # 后序遍历的最后一个节点为根节点
        left = 0
        while sequence[left] < root:
            left += 1
        # 此时left取得右子树的第一个坐标
        for j in range(left,length-1): # 对右子树做循环，因为之前的while循环已经核对了左子树的值均小于root
            if sequence[j]<root:
                return False
        # 对每一棵子树进行递归操作，左子树 and 右子树
        return self.VerifySquenceOfBST1(sequence[:left]) and self.VerifySquenceOfBST1(sequence[left:length-1])
        # 为啥这里要用or？？？？？？不应该用and吗？？？
        # 4 6 7 5 应该返回true的，但由于右边最后一轮切分的left=1，length=2，右边切分的长度为0，会返回false，所以
        # and之后就变为了False。如果要用and的话，那么把length == 0，return的改为True吧。但输入为[]的时候会报错！！
        # 所以我选择另起炉灶！
        # 前序序列或中序序列只需要改变root的位置值，以及最后那两个递归的左子树右子树位置即可


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
        left = self.FindPath(root.left,expectNumber-root.val) # 在左子树里找
        right = self.FindPath(root.right,expectNumber-root.val) # 也在右子树里找
        res = []
        for i in left+right: # 在left与right子树构成的一个集合中寻找i
            res.append([root.val]+i)   # [5]+[6]就是[5,6],这样好像并不能保证数组长度大的靠前，不过能通过
        return res                     # [5,6]+[8] = [5,6,8]
        # 不断返回上次递归的栈中，与对应的根节点值累加

# 36、二叉搜索树与双向链表
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
        # 对剩下的结点按顺序进行操作
        for i in range(1, len(res) - 1):
            res[i].left = res[i - 1]
            res[i].right = res[i + 1]
        return res[0] # 返回排好序的双向链表的头结点


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

    # 反序列化，没看。。。
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

# 54、二叉搜索树的第K大节点
"""
给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。
# 按照中序遍历的顺序遍历一棵二叉搜索树，则遍历序列的数值是递增排序的，然后直接取出索引为[k-1]的值即可
"""

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
        # 左，根，右
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
        return self.recur(pRoot,1) # 非空树，深度初始值为1

    def recur(self,root,level):
        maxlevel = level
        if root.left != None:
           maxlevel = max(maxlevel,self.recur(root.left,level+1)) # 注意，每进入一层递归，送入的深度值都要+1
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
        right_depth = self.getDepth(pRoot.right) # 这样就从最左边的子树开始判断，并逐步回溯
        if abs(left_depth-right_depth) > 1:
            self.__is_balanced = False
            # 一旦任意一个子树出现这种情况，立即置于false
            return 0 # 随便返回一个，目的只在于一步步退出堆栈
        return left_depth+1 if left_depth>right_depth else right_depth+1 # 每向上返回一步堆栈，树的长度就增加1
    def IsBalanced_Solution(self, pRoot):
        self.getDepth(pRoot)
        return self.__is_balanced


# 68、树中两个节点的最低公共祖先
"""
是二叉树，并且是二叉搜索树：
思路：
二叉搜索树是经过排序的，位于左子树的节点都比父节点小，位于右子树的节点都比父节点大。
既然要找最低的公共祖先节点，我们可以从根节点开始进行比较。
若当前节点的值比两个节点的值都大，那么最低的祖先节点一定在当前节点的左子树中，则遍历当前节点的左子节点；
反之，若当前节点的值比两个节点的值都小，那么最低的祖先节点一定在当前节点的右子树中，则遍历当前节点的右子节点；
这样，直到找到一个节点，位于两个节点值的中间，则找到了最低的公共祖先节点。

如果存在parent指针，则分别从输入的p节点和q节点指向root根节点（倒序了），其实这就是两个单链表。
问题转化为求两个单链表相交的第一个公共节点（先求链表长度之差，然后一步一步走，直到走到第一个公共节点）

那如果不存在parent指针呢？
用两个链表分别保存从根节点到输入的两个节点的路径，然后把问题转换成两个链表的最后公共节点。（简单）

# 1
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

# 3
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
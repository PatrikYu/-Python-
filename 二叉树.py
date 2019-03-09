定义树结构

class Tree(object):
    def __init__(self,val):
        self.val = val
        self.right = None
        self.left = None

前序遍历

def pre_iter(node):
    if not node:
        return
    print(node.val)
    pre_iter(node.left)
    pre_iter(node.right)

中序遍历

def mid_iter(node):
    if not node:
        return
    mid_iter(node.left)
    print(node.val)
    mid_iter(node.right)

后序遍历

def post_iter(node):
    if not node:
        return
    post_iter(node.left)
    post_iter(node.right)
    print(node.val)

代码虽简单,但是个人认为树结构是学习递归,乃至学习函数式编程的好例子.宜细细品味,挖个函数式编程的坑,希望以后自己可以写出一篇完整的文章.

根据先序遍历和中序遍历,重建二叉树 牛客网-剑指offer 重建二叉树

class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        if len(tin) == 0:
            return None
        else:
            #先序序列当中第一个值必定为该层父节点
            root = TreeNode(pre[0])
            #root已经包含了返回的节点值
            slt = tin.index(pre[0])
            root.left = self.reConstructBinaryTree(pre[1:1+slt],tin[:slt])
            root.right = self.reConstructBinaryTree(pre[1+slt:],tin[slt+1:])
        return root

从后序遍历和中序遍历重建二叉树,来自LeetCode
106. Construct Binary Tree from Inorder and Postorder Traversal

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not inorder and not postorder:
            return None
        else:
            val = postorder.pop()
            root = TreeNode(val)
            i = inorder.index(val)
            root.left = self.buildTree(inorder[:i],postorder[:i])
            root.right = self.buildTree(inorder[i+1:],postorder[i:])
            return root

二叉树的深度优先遍历(DFS)

def dfs(self, root):
    stk = []#以栈结构实现
    if not root:
        return root
    else:
        stk.append(root)
        while stk:
            node = stk.pop()
            print(node.val)#这里可执行一切使用当前结点的方法
            #右子结点先入栈
            if node.right:
                stk.append(node.right)
            #左子结点后入栈
            if node.left:
                stk.append(node.left)

二叉树的广度优先遍历(BFS)

def bfs(self, root):
    q = []#以队列结构实现
    if not root:
        return root
    else:
        q.append(root)
        while q:
            #以list结构实现栈与队列结构知识pop()与pop(0)的区别
            node = q.pop(0)
            print(node.val)#这里可执行一切使用当前结点的方法
            #左子结点先入栈
            if node.left:
                q.append(node.left)
            #右子结点后入栈
            if node.right:
                q.append(node.right)

在迭代实现遍历时,深度优先以栈结构实现,广度优先以队列结构实现.

附:二叉树的深度优先遍历递归实现

实际上树的前序后序中序遍历都是深度优先的具体情况,以上面前序遍历的代码即可作为树的深度优先遍历的递归实现.

def dfs_rcv(node):
    if not node:
        return
    print(node.val)
    dfs_rcv(node.left)
    dfs_rcv(node.right)

二叉树的镜像(对称反转一个二叉树)

def Mirror(root):
    if not root:
        return
    else:
        root.left, root.right = Mirror(root.right), Mirror(root.left)
        #错误写法
        #root.left = Mirror(root.right)
        #root.right = Mirror(root.left)
        return root

二叉树的深度

    def depth(root):
        if not root:
            return 0
        else:
            nleft = depth(root.left)
            nright = depth(root.right)
            return max(nleft, nright) + 1

判断是否平衡二叉树

class Solution:
    def getDepth(self, root):
        if not root:
            return 0
        else:
            left = self.getDepth(root.left)
            right = self.getDepth(root.right)
            return max(left,right) + 1

    def IsBalanced_Solution(self, pRoot):
        # write code here
        if not pRoot:
            return True
        else:
            left = self.getDepth(pRoot.left)
            right = self.getDepth(pRoot.right)
            if abs(left-right) <= 1:
                return True
            else:
                return False

检验两棵树是否相同(LeetCode:SameTree)

        if not p and not q:
            return True
        else:
            if not p or not q:
                return False
            else:
                return (p.val == q.val) and self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)

判断对称二叉树

注意: 给出case当中,空树也为True.左右子树为空,也为True.

class Solution:
    def issame(self,p1,p2):
        if not p1 and not p2:
            return True
        else:
            if p1 and p2:
                return (p1.val == p2.val) and self.issame(p1.left,p2.right) and self.issame(p1.right,p2.left)
            else:
                return False

    def isSymmetrical(self, pRoot):
        # write code here
        if not pRoot:
            return True
        else:
            if pRoot.left and pRoot.right:
                return self.issame(pRoot.left, pRoot.right)
            else:
                return True

[牛客网]二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

class Solution:
    def Convert(self, pRootOfTree):
        # 仅针对测试用例为空根节点
        if not pRootOfTree:
            return pRootOfTree
        # 叶子结点返回结点
        if not pRootOfTree.left and not pRootOfTree.right:
            return pRootOfTree
        else:
        #针对所有非叶节点
            if pRootOfTree.left:
                lch = self.Convert(pRootOfTree.left)
            #因为下一层的return值为最左端结点
            #因此作为根节点的连接点,左子树构造的链表需遍历到最右端
                while lch.right:
                    lch = lch.right
                pRootOfTree.left = lch
                lch.right = pRootOfTree
            if pRootOfTree.right:
                rch = self.Convert(pRootOfTree.right)
            #下一层的return值为最左端结点
            #因此对于根节点来说,直接连接即可
                pRootOfTree.right = rch
                rch.left = pRootOfTree
            head = pRootOfTree
            #每层构造完链表,返回最左端,也就是头结点
            while head.left:
                head = head.left
            return head

1) 分析每一层的任务:构造一个 “[左子树构造的链表]<–>根节点<–>[右子树构造的链表]”类似这样结构的链表
2) 递归问题的性质:问题的返回值要求决定了每次递归的返回值.
在这个问题当中,要求返回构造后的双向链表的头结点,即1)当中[左子树构造的链表]的头结点.
如果是只要求返回根节点在链表中的位置,问题会简单一些,代码会更易读一点.
3) 确定递归终止条件:
如果是[没有左子结点 or 没有右子节点]作为终止条件,会导致不能访问到所有节点.因此应该选择[没有左子结点 and 没有右子节点]作为终止条件.
4) 到这里solution已经得出大体结构,编码试错.发现提示访问了Nonetype的right或者left,加入if pRootOfTree.right 和 if pRootOfTree.left: 解决.

LeetCode 117. Populating Next Right Pointers in Each Node II

Given the following binary tree,

     1
   /  \
  2    3
 / \    \
4   5    7

After calling your function, the tree should look like:

    1 -> NULL
   /  \
  2 -> 3 -> NULL
 / \    \
4-> 5 -> 7 -> NULL

# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if not root:
            return root
        else:
            layer = [root]
            while layer:
                next_layer = []
                p = head = layer.pop(0)
                if p.left:
                    next_layer.append(p.left)
                if p.right:
                    next_layer.append(p.right)
                while layer:
                    node = layer.pop(0)
                    if node.left:
                        next_layer.append(node.left)
                    if node.right:
                        next_layer.append(node.right)
                    p.next = node
                    p = p.next
                layer = next_layer

广度优先遍历的一种形式.

二叉树的下一个结点

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return pNode
        else:
            if pNode.right:
                pNode = pNode.right
                while pNode.left:
                    pNode = pNode.left
                return pNode
            else:
                p = pNode
                while p.next:
                    pp = p.next
                    if pp.left == p:
                        return pp
                    p = pp
                return p.next


思路:
1) 对于给定结点pNode, 分为if pNode.right else 有无右子节点两种情况;
2) pNode有右结点 则中序遍历的下一个结点必定在右子树当中,具体位置是右子树的最左结点;
3) pNode无右结点 则往父结点寻找 从底往上 当找到父节点为祖父结点的左结点时 该祖父结点则为返回结点值;
4) 若找不到3) 中的结构,则说明当前结点pNode 为中序遍历顺序的最右端.

112. Path Sum
查找是否存在从根到叶的路径path, path上的结点的值相加为目标值sum. 存在返回True 不存在返回False

class Solution:
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            return False
        else:
            if root.left and root.right:
                return self.hasPathSum(root.left,sum-root.val) or self.hasPathSum(root.right,sum-root.val)
            elif root.left:
                return self.hasPathSum(root.left,sum-root.val)
            elif root.right:
                return self.hasPathSum(root.right,sum-root.val)
            else:
                return bool(root.val == sum)
其实是因为对于输入为([],0),即空树和0的特例, LeetCode要求返回False.如果该特例可以返回True,有以下更简洁的写法:

class Solution:
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            return bool(sum==0)
        else:
            return self.hasPathSum(root.left,sum-root.val) or self.hasPathSum(root.right,sum-root.val)

Leetcode 113. Path Sum II
查找是否存在从根到叶的路径path, path上的结点的值相加为目标值sum. 以二维数组形式返回所有符合的路径

class Solution:
    def pathSum(self, root, sumk):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        if not root:
            return res
        stk = [(root,[root.val])]
        while stk:
            node,path = stk.pop()
            if node.left or node.right:
                if node.left:
                    lp = path + [node.left.val]
                    nxt = (node.left,lp)
                    stk.append(nxt)
                if node.right:
                    rp = path + [node.right.val]
                    nxt = (node.right,rp)
                    stk.append(nxt)
            else:
                if sum(path) == sumk:
                    res.append(path)
        return res
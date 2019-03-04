import sys
reload(sys)
sys.setdefaultencoding('utf8')


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

# 18、删除链表中重复节点
"""
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:  # 空链表或单链表
            return pHead
        head1 = pHead.next  # 取得第二个指针（指向第二个结点）
        if head1.val != pHead.val: # 不是重复元素，将下一个结点的表头变量传入，继续查找重复元素
            pHead.next = self.deleteDuplication(head1) # 第二个结点传入
        else:
            while head1.val == pHead.val and head1.next is not None:  # 是重复元素,且head1不是最后一个结点
                head1 = head1.next # 舍弃了第二个结点，直接连到了第三个结点上，继续循环，直到相邻的没有重复元素或只剩下两个重复的结点
            if head1.val != pHead.val: # 相邻的重复元素清理好了
                pHead = self.deleteDuplication(head1) # 依然是第二个结点传入，但是同时删除之前曾经重复过的一个结点（因为重复的结点一个也不保留）
            else: # 还剩下两个重复的结点，全删除，返回None
                return None
        return pHead

# 19、正则表达式匹配（没有完全搞懂）
"""
请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 
在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
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
        if (len(pattern) > 1 and pattern[1] == '*'):# s不为空且第一个字符匹配
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

# 21、表示数值的字符串
"""
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
"""
"""
笑死我了，用python写，能转为浮点数类型的就是数值，转换不了的就表示不了数值哈哈没毛病
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
使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
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

# 22、链表中倒数第k个节点
"""
输入一个链表，输出该链表中倒数第k个结点。
即正着数第n-k+1个结点,但链表的长度n未知
先把链表中所有结点（指针）放到list里直接取.注意k不能超过len(l)也不能小于1.直接l[-k]
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        l=[]
        while head:
            l.append(head)
            head = head.next
        if k>len(l) or k<1:
            return
        return l[len(l)-k]   # 倒数第k个就是 l[-k]

# 23、链表中环的入口结点
"""
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        # 思路：遍历链表，若环存在，遍历遇见第一个重复的即为入口结点
        # 将链表中每个指针都存入一个list中，若遍历过程中，发现有相同的指针，立即返回
        list1=list()
        p = pHead
        while p:
            if p in list1:
                return p
            else:
                list1.append(p)
            p=p.next
        return None

# 24、反转链表
"""
输入一个链表，反转链表后，输出新链表的表头。
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        # 如果是空链表或单链表，直接返回pHead即可
        if pHead==None or pHead.next==None:
            return pHead
        # 第一个结点pHead的next变为None，第二个结点的next变为第一个结点
        # 但是第一个结点的next变为None后，第二个结点就取不出来了，因此先将第二个结点的指针放到temp中
        last = None # 原本顺序的上一个结点，原来的表头的上一个结点为None
        while pHead:
            tmp = pHead.next
            pHead.next = last
            last = pHead
            pHead = tmp
        # 最后一次循环时，pHead.next == None,而last=pHead，因此最后返回last
        return last

# 25、合并两个排序的链表
"""
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。(记得考虑空链表)
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        # 思路：每次比较第一个结点元素值的大小，谁小谁放前面，然后另外一个不变，继续比较
        p = ListNode(1)   # 定义一个第一个结点值为1的单链表，合成链表的头结点
        first = p # 记录下合成的p链表的表头指针，first.next就是最后的输出链表
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val:
                p.next = pHead1
                pHead1 = pHead1.next
            else:
                p.next = pHead2
                pHead2 = pHead2.next
            # 注意！！！此时p指针还是指在同一个地方！！应该让它向后指一步！！！！
            p = p.next
        # 当有任意链表为空时，必有一链表没空，把没空的那个链表用next衔接上就可以了
        if pHead1:
            p.next = pHead1
        elif pHead2:
            p.next = pHead2
        return first.next

# 26、树的子结构
"""
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
思路：第一步，在树A中找到和树B的根结点的值一样的结点R（写个循环）
      第二步，判断R的左子树和右子树是否和B相同（写个递归，增加个新函数 is_subtree ）
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

    def is_subtree(self,A,B):
        if B is None:
            return True
        if A is None or A.val != B.val: # A的某子树的所有节点都遍历完了 或 此两个节点值不同
            return False
        # 在树A中找到了和树B的根结点的值一样的结点R
        return self.is_subtree(A.left,B.left) and self.is_subtree(A.right,B.right)





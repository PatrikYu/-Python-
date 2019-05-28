# 6、从尾到头打印链表
"""
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
思路：第一个遍历到的节点最后一个输出，而最后一个遍历到的节点第一个输出，这就是典型的"后进先出"，可以用一个栈来实现这种顺序：
每经过一个节点的时候，把该节点放到一个栈中，当遍历完整个链表后，再从栈顶开始逐个输出节点的值。
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
            l.append(head.val)
            head = head.next
        return l[::-1]

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
        # 若phead接着的不是重复节点，将phead的下个节点送入递归，处理接下来的片段，并将pHead.next指向head1
        head1 = pHead.next
        if head1.val != pHead.val:
            pHead.next = self.deleteDuplication(head1)
        # 是重复元素
        else:
            while head1.val == pHead.val and head1.next is not None:  # 是重复元素,且head1不是最后一个结点
                head1 = head1.next # 舍弃了第二个结点，直接连到了第三个结点上，
                                   # 继续循环，直到相邻的没有重复元素或下一个重复的节点是最后一个节点
            if head1.val != pHead.val: # 相邻的重复元素清理好了
                pHead = self.deleteDuplication(head1) # 依然是第二个结点传入，但是同时删除之前曾经重复过的一个结点
                                                      # （因为重复的结点一个也不保留）
            else: # 若两个节点均重复，全删除，返回None
                return None
        return pHead

# 22、链表中倒数第k个节点
"""
输入一个链表，输出该链表中倒数第k个结点。
思路：为了实现只遍历一次就能找到倒数第k个节点，可以定义两个指针：第一个指针从链表的头指针开始遍历向前走k-1步，
第二个指针保持不动；从第k步开始，第二个指针也开始从链表的头指针开始遍历。由于两个指针的距离保持在k-1，当第一个指针
到达链表的尾节点时，第二个指针正好指向倒数第k个节点。
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        if not head or not k:
            return None
        left, right = head, head
        for i in range(k - 1):
            if not right.next:
                return None
            right = right.next
        while right.next:
            left = left.next
            right = right.next
        return left

# 23、链表中环的入口结点
"""
思路：
1、是否存在环：定义两个指针，一个指针一次走一步，另一个指针一次走两步，如果走得快的指针追上了走得慢的指针，即二者相遇，
那么包含环，两个指针相遇的节点一定是在环中，可以从这个节点出发，走得慢的指针一边继续向前移动一边计数，当再次回到这个节点，
就可以得到环中节点数n。若走得快的指针走到了链表的末尾(m_pNext指向NULL)都没有与第一个指针相遇，那么链表就不包含环。

2、如何找到环的入口：先定义两个指针P1，P2指向链表的头节点，若链表中的环有n个节点，则指针P1先在链表上向前移动n步，
然后两个指针以相同的速度向前移动。当第二个指针指向环的入口节点时，第一个指针已经围绕着环走了一圈，又回到了入口节点。
即他们相遇的点就是环的入口节点。

"""
"""
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
        # 思路：遍历链表，若环存在，遍历遇见第一个重复的即为入口结点
        # 将链表中每个指针都存入一个list中，若遍历过程中，发现有相同的指针，立即返回
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        list1=list()
        p = pHead
        while p:
            if p in list1:
                return p
            else:
                list1.append(p)
            p = p.next
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

# 单链表的排序算法
# 首先考虑基于移动元素的单链表排序算法
# 原理：确保crt左边的所有元素都比它小，若比它大，则调换顺序；而移动的指针p左边的一定都是有序的
def sort1(self):
    if self._head is None:
        return
    crt = self._head.next # 从第二个结点开始处理
    while crt:
        p = self._head # 取得首结点
        x = crt.elem
        while p is not crt and p.elem <= x: # 若crt左边都比他小，p指针向右移一步
            p = p.next
        while p is not crt: # 出现一个比它大的，交换值，且p指针向右移一步（就一轮交换值）
            y = p.elem
            p.elem = x
            p = p.next
            crt.elem = y
            # x = y  # crt.elem = x 此操作在下一步中进行，我觉得这样没必要
        # crt.elem = x   #
        crt = crt.next
"""
用堆排序实现 O(nlogn)的链表排序：
https://blog.csdn.net/qq_34364995/article/details/80994110
"""
# 25、合并两个排序的链表
"""
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。(记得考虑空链表)
        # 思路：每次比较第一个结点元素值的大小，谁小谁放前面，然后另外一个不变，继续比较
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

# 52、两个链表的第一个公共节点
"""
输入两个链表，找出它们的第一个公共结点。
"""
"""
链表有公共结点，意味着这两个链表从某一节点开始，他们的.next都指向同一个节点，由于是同一个节点，后面所有的节点也必然相同
下面的实现：先把一链表的所有节点放到一个list中，然后遍历第二个链表看看有没有节点在list1中。
这样的复杂度是O(m*n)。复杂度更低的方法：只要找到第一个相同的节点就好，可以先分别遍历一遍两个链表，得到长度之差L，先让长的链表
向后走L步。比如差为1，就走到第二个节点处。接下来两个链表分别从第1个和第2个节点出发，直到找到第一个相同的节点，返回即可。
这样做的复杂度为O(m+n)
"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        list1 = []
        node1 = pHead1
        node2 = pHead2
        while node1:
            list1.append(node1.val)
            node1 = node1.next
        while node2:
            if node2.val in list1:
                return node2
            else:
                node2 = node2.next
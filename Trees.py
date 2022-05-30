#!/usr/bin/env python
# coding: utf-8

# # TREES

# In[ ]:


'''
Edges = nodes-1
Depth of x = Length of path from root to x
Height of x = Number of edges in longest path from x to leaf
'''


# In[1]:


class Node:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right=None
    def __str__(self): #str representation of the object of class
        return str(self.data)


# In[2]:


root=Node(40)
print(root.left)
print(root.right)


# In[3]:


left_child=Node(20)
right_child=Node(56)


# In[4]:


root.left=left_child
root.right=right_child


# In[5]:


print(root)
print(root.left)
print(root.right)


# # BINARY SEARCH TREES (BST)

# In[ ]:


'''
left side is less than equal to parent nodes and root node
right side is greater than parent nodes and root node
'''

'''
TRAVERSAL IN BINARY SEARCH TREE
.Coming or going or reaching all the nodes which are present in the tree
.It can be done in two ways-
a>Breath first
b>Depth first
    1>In order - .Covering all the nodes from :-
                  Left subtree -> current data -> right subtree
        EXAMPLE 1-
                    A
                  /   \
                  B    C
                 /      \
                 D       I
                /\        \
                E F        J
                  /\       /\
                  G H      K L
                ANSWER- EDGFHBACIKJL
                
     EXAMPLE 2-
                A
                /\
                B C
                   /\
                   D E
                     /
                     E
                     /
                     G
                     
                ANSWER- BADCGEE
                  
    2>Pre order -> ROOT -> LEFT -> RIGHT
                
    3>Post order -> LEFT -> RIGHT -> ROOT
    
        
'''


# In[1]:


class Node():
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right=None


# In[5]:


#Recurion will be used here and In recursion number of lines of codes is less
#but it is generally difficult to uinnderstand.
def insert(node,data):
    #Exit condition (when the considered subtree is empty)
    if node is None:
        return Node(data)
    #when the subtree is not empty
    if data<=node.data:
        node.left=insert(node.left,data)
    else:
        node.right=insert(node.right, data)
    return node


# In[6]:


root=Node(100)


# In[7]:


insert(root,50)


# In[8]:


root.left.data


# In[9]:


insert(root,120)


# In[10]:


root.right.data


# In[11]:


insert(root,70)


# In[12]:


root.left.right.data


# In[13]:


insert(root,110)


# In[14]:


root.right.left.data


# In[15]:


#Depth First Traversal


# In[17]:


#recursion based
#inorder gives you sorted sequense
def inorder(root):
    if root is not None:
        inorder(root.left)
        print(root.data)
        inorder(root.right)


# In[18]:


inorder(root)


# In[19]:


def preorder(root):
    if root is not None:
        print(root.data)
        inorder(root.left)
        inorder(root.right)


# In[24]:


def postorder(root):
    if root is not None:
        inorder(root.left)
        inorder(root.right)
        print(root.data)


# In[21]:


preorder(root)


# In[25]:


postorder(root)


# In[26]:


#Breadth First Traversal


# In[29]:


def bfs(root):
    queue=[]
    
    #exit  condition
    if root is None:
        return
    queue.append(root)
    
    while(len(queue)>0):
        popped_node=queue.pop(0)
        print(popped_node.data)
        if popped_node.left is not None:
            queue.append(popped_node.left)
        if popped_node.right is not None:
            queue.append(popped_node.right)


# In[30]:


bfs(root)


# In[ ]:





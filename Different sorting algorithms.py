#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Binary Search 
#Assuming the list is sorted
#mid= low+(high_low)//2

def binary_search(A,low,high,key):
    if high>=low:
        mid= low+(high-low)//2
        if A[mid]==key:
            return mid
        elif key>A[mid]:
            return binary_search(A,mid+1,high,key)
        else:
            return binary_search(A,low,mid-1,key)
    else:
        return 'not found'


# In[3]:


#Quick sort
def partition(A,l,r):
    x=A[l]
    j=l
    for i in range(l+1,r+1):
        if A[i]<=x:
            j+=1
            A[j],A[i]=A[i],A[j]
    A[l],A[j]=A[j],A[l]
    return j
def quicksort(A,l,r):
    if l>=r:
        return 
    m=partition(A,l,r)
    quicksort(A,l,m-1)
    quicksort(A,m+1,r)
    return A
A=[34,6,2,3,33,9,9,-5,6,1]
quicksort(A,0,len(A)-1)


# In[4]:


#Merge Sort
def Merge(B,C):
    D=[]
    while len(B)!=0 and len(C)!=0:
        b=B[0]
        c=C[0]
        if b<=c:
            D.append(b)
            B.pop(0)
        else:
            D.append(c)
            C.pop(0)
    if len(B)!=0:
        D.extend(B)
    if len(C)!=0:
        D.extend(C)
    return D
def merge_sort(A):
    n=len(A)
    if n==1:
        return A
    m=n//2
    print(A[:m],'*',A[m:])
    B=merge_sort(A[:m])
    C=merge_sort(A[m:])
    A_dash=Merge(B,C)
    print(A_dash)
    return A_dash
x=[30,-11,40,20,13,63,75,101]
merge_sort(x)


# In[1]:


#Insertion Sort 
list=[10,1,5,0,6,8,7,3,11,4]

i=1
while(i<10):
  element=list[i]
  j=i
  i=i+1

  while(j>0 and list[j-1]>element):
    list[j]=list[j-1]
    j=j-1

  list[j]=element

i=0
while(i<10):
  print (list[i])
  i=i+1


# In[ ]:





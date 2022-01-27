#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Q1 Write a program to find all pairs of an integer array whose sum is equal to a given number
def printPairs(arr, n, sum):
    for i in range(0, n ):
        for j in range(i + 1, n ):
            if (arr[i] + arr[j] == sum):
                print("(", arr[i],",",arr[j],")", sep = "")

arr = [1, 5, 7, -1, 5]
n = len(arr)
sum = 6
printPairs(arr, n, sum)


# In[8]:


#Q2 Write a program to reverse an array in place? In place means you cannot create a new array. You have to update the original array.
def reverseList(arr,start,end):
    while start<end:
        arr[start],arr[end]=arr[end],arr[start]
        start+=1
        end-=1
arr = [1, 2, 3, 4, 5]
reverseList(arr, 0, 4)
print("Reversed list is")
print(arr)


# In[15]:


#Q3 Write a program to check if two strings are a rotation of each other python
def areRotations(string1, string2):
    size1 = len(string1)
    size2 = len(string2)
    temp = ''
    if size1 != size2:
        return 0
    temp = string1 + string1
    if (temp.count(string2)> 0):
        return 1
    else:
        return 0
string1 = "ABCD"
string2 = "CDAB"

if areRotations(string1, string2):
    print ("Strings are rotations of each other")
else:
    print ("Strings are not rotations of each other")


# In[34]:


#Q4 Write a program to print the first non-repeated character from a string?
def first_non_repeating_character(str1):
  char_order = []
  ctr = {}
  for c in str1:
    if c in ctr:
      ctr[c] += 1
    else:
      ctr[c] = 1 
      char_order.append(c)
  for c in char_order:
    if ctr[c] == 1:
      return c
  return None

print(first_non_repeating_character('abcdef'))
print(first_non_repeating_character('avanish'))
print(first_non_repeating_character('abcabcdef'))
print(first_non_repeating_character('aabbcc'))


# In[36]:


#Q5 Write a program to implement it
def TowerOfHanoi(n , source, destination, auxiliary):
    if n==1:
        print ("Move disk 1 from source",source,"to destination",destination)
        return
    TowerOfHanoi(n-1, source, auxiliary, destination)
    print ("Move disk",n,"from source",source,"to destination",destination)
    TowerOfHanoi(n-1, auxiliary, destination, source)
n = 4
TowerOfHanoi(n,'A','B','C')
# A, C, B are the name of rods


# In[37]:


#Q6 Write a program to convert postfix to prefix expression.
def isOperator(x):
	if x == "+":
		return True

	if x == "-":
		return True

	if x == "/":
		return True

	if x == "*":
		return True

	return False
def postToPre(post_exp):
	s = []
	length = len(post_exp)
	for i in range(length):
		if (isOperator(post_exp[i])):
			op1 = s[-1]
			s.pop()
			op2 = s[-1]
			s.pop()
			temp = post_exp[i] + op2 + op1
			s.append(temp)
		else:
			s.append(post_exp[i])
	ans = ""
	for i in s:
		ans += i
	return ans
if __name__ == "__main__":

	post_exp = "AB+CD-"
	print("Prefix : ", postToPre(post_exp))


# In[38]:


#Q7 Program to convert prefix to Infix
def prefixToInfix(prefix):
	stack = []
	i = len(prefix) - 1
	while i >= 0:
		if not isOperator(prefix[i]):
			stack.append(prefix[i])
			i -= 1
		else:
			str = "(" + stack.pop() + prefix[i] + stack.pop() + ")"
			stack.append(str)
			i -= 1
	
	return stack.pop()

def isOperator(c):
	if c == "*" or c == "+" or c == "-" or c == "/" or c == "^" or c == "(" or c == ")":
		return True
	else:
		return False
if __name__=="__main__":
	str = "*-A/BC-/AKL"
	print(prefixToInfix(str))


# In[47]:


# Q8 Write a program to check if all the brackets are closed in a given code snippet
def areBracketsBalanced(expr):
	stack = []
	for char in expr:
		if char in ["(", "{", "["]:
			stack.append(char)
		else:
			if not stack:
				return False
			current_char = stack.pop()
			if current_char == '(':
				if char != ")":
					return False
			if current_char == '{':
				if char != "}":
					return False
			if current_char == '[':
				if char != "]":
					return False
	if stack:
		return False
	return True
if __name__ == "__main__":
	expr = "{()}[]"
	if areBracketsBalanced(expr):
		print("Balanced")
	else:
		print("Not Balanced")


# In[55]:


#Q9 Write a program to reverse a stack
class stack:
    def __init__(self):
        self.stack=[]
        
    def push(self,element):
        self.stack.append(element)
        return f"Element {element} Pushed"
    
    def is_empty(self):
        if len(self.stack)==0:
            return True
        else:
            return False

    def display(self):
        for i in self.stack:
            print(i)
            
my_stack=stack()
print(my_stack.push(1))
print(my_stack.push(2))
print(my_stack.push(3))
my_stack.display()


# In[56]:


#Q10 Write a program to find the smallest number using a stack
stack=[1,2,3]
print(min(stack))S


# In[ ]:





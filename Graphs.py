#!/usr/bin/env python
# coding: utf-8

# # Adjancy List representation

# In[1]:


graph_1={
    "A":["B","C","D","E"],
    "B":["A"],
    "C":["A","D"],
    "D":["A","C"]
}


# In[2]:


class graph:
    def __init__(self):
        self.data = {}
        
    def addEdge(self, u , v):# u = edges, v = vertices
        if u in self.data.keys():
            self.data[u].append(v)
        else:
            self.data[u] = [v]#key value pair
        if v not in self.data.keys():
            self.data[v] = []
    
    #Breadth First Traversal
    # Step1:append to queue and mark it as visited
    # Step2:pop from front 
    def bfs(self):
        visited = []
        queue = []
        for v in self.data.keys():
            if v in visited:
                continue
            queue.append(v)
            visited.append(v)
            while len(queue) > 0:
                popped_item = queue.pop(0)
                print(popped_item,end='->')
                for neighbour in self.data[popped_item]:
                    if neighbour not in visited:
                        queue.append(neighbour)
                        visited.append(neighbour)
    # A function used by DFS
    def DFSUtil(self, v, visited):
  
        # Mark the current node as visited and print it
        visited[v]= True
        print v,
  
        # Recur for all the vertices adjacent to
        # this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.DFSUtil(i, visited)
  
  
    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self):
        V = len(self.graph)  #total vertices
  
        # Mark all the vertices as not visited
        visited =[False]*(V)
  
        # Call the recursive helper function to print
        # DFS traversal starting from all vertices one
        # by one
        for i in range(V):
            if visited[i] == False:
                self.DFSUtil(i, visited)
                
    def BFS(s, l):
     
    V = 100
     
    # Mark all the vertices
    # as not visited
    visited = [False] * V
    level = [0] * V
  
    for i in range(V):
        visited[i] = False
        level[i] = 0
  
    # Create a queue for BFS
    queue = deque()
  
    # Mark the current node as
    # visited and enqueue it
    visited[s] = True
    queue.append(s)
    level[s] = 0
  
    while (len(queue) > 0):
         
        # Dequeue a vertex from
        # queue and print
        s = queue.popleft()
        #queue.pop_front()
  
        # Get all adjacent vertices
        # of the dequeued vertex s.
        # If a adjacent has not been
        # visited, then mark it
        # visited and enqueue it
        for i in adj[s]:
            if (not visited[i]):
  
                # Setting the level
                # of each node with
                # an increment in the
                # level of parent node
                level[i] = level[s] + 1
                visited[i] = True
                queue.append(i)
  
    count = 0
    for i in range(V):
        if (level[i] == l):
            count += 1
             
    return count

    # A utility function to do DFS of graph
# recursively from a given vertex u.
    def DFSUtill(u, adj, visited):
        visited[u] = True
        for i in range(len(adj[u])):
            if (visited[adj[u][i]] == False):
                DFSUtil(adj[u][i], adj, visited)

    # Returns count of tree is the
    # forest given as adjacency list.
    def countTrees(adj, V):
        visited = [False] * V
        res = 0
        for u in range(V):
            if (visited[u] == False):
                DFSUtill(u, adj, visited)
                res += 1
        return res
    
    def isCyclicUtil(self, v, visited, recStack):
 
        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True
 
        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.isCyclicUtil(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True
 
        # The node needs to be popped from
        # recursion stack before function ends
        recStack[v] = False
        return False
 
    # Returns true if graph is cyclic else false
    def isCyclic(self):
        visited = [False] * (self.V + 1)
        recStack = [False] * (self.V + 1)
        for node in range(self.V):
            if visited[node] == False:
                if self.isCyclicUtil(node,visited,recStack) == True:
                    return True
        return False


# In[3]:


x=graph()


# In[4]:


x.addEdge('A','B')
x.addEdge('A','C')
x.addEdge('A','D')
x.addEdge('B','A')
x.addEdge('C','A')
x.addEdge('C','D')
x.addEdge('D','A')
x.addEdge('D','C')


# In[5]:


x.data


# In[6]:


x.bfs()


# In[ ]:


x.DFS()


# In[7]:


y=graph()


# In[8]:


y.addEdge('0','1')
y.addEdge('1','2')
y.addEdge('2','0')
y.addEdge('1','3')
y.addEdge('3','4')
y.addEdge('4','5')
y.addEdge('2','5')


# In[9]:


y.data


# In[10]:


y.bfs()


# In[ ]:


if y.isCyclic() == 1:
    print "Graph has a cycle"
else:
    print "Graph has no cycle"


**Breadth-First Search Implementation**

**Description:**
- BFS explores all vertices at the current depth level before moving on to the next depth level.
- It uses a queue to keep track of the next vertex to explore.
- BFS is generally used for exploring graphs in a 'level-wise' manner, where you explore nodes layer by layer from the start node.
- It's best suited for finding the shortest path in an unweighted graph

```python
from collections import deque

def bfs(graph, start_vertex):
	visited = set()
	queue = deque([start_vertex])
	bfs_order = []

	while queue:
		vertex = queue.popleft()
		if vertex not in  visited:
			visited.add(vertex)
			bfs_order.append(vertex)

			for neighbor in graph[vertex]:
				if neighbor not in visited:
					queue.append(neighbor)
	return bfs_order
```

Complexities
*Time Complexity:*
- $O(V + E)$ where V is the number of vertices and E is the number of edges.
- In the worst case, we visit every vertex and every edge

*Space Complexity*
- $O(V)$ for the queue and the set to track visited vertices.

Applications
- Shortest path in unweighted graphs
- Level order traversal
- Finding connected components
- Web crawlers
- Shorted path in unweighted grid

Disadvantages
- Requires more memory, especially in dense graphs, as it stores all vertices at the current level in the queue
- May not be efficient for graphs with a large branching factor



**Depth-First Search Implementation**

**Description:**
- DFS explores as far as possible along each branch before backtracking.
- It uses a stack (or recursion) to explore each vertex.
- DFS can be implemented using recursion or an explicit stack.
- DFS is often used for exploring the entire graph or for finding paths in directed graphs.

```python
def dfs_recursive(graph, vertex, visited, dfs_order):
	visited.add(vertex)
	dfs_order.append(vertex)

	for neighbor in graph[vertex]:
		if neighbor not in visited:
			dfs_recursive(graph, neighbor, visited, dfs_order)

def dfs(graph, start_vertex):
	visited = set()
	dfs_order = []
	dfs_recursive(graph, start_vertex, visited, dfs_order)
	return dfs_order
```

```python
def dfs_iterative(graph, start_vertex):
	visited = set()
	stack = [start_vertex]
	dfs_order = []

	while stack:
		vertex = stack.pop

		if vertex not in visited:
			visited.add(vertex)
			dfs_order.append(vertex)

			for neighbor in reversed(graph[vertex]):
				if neighbor not in visited:
					stack.append(neighbor)
	return dfs_order
```

Complexities
*Time Complexity*
- $O(V + E)$ where $V = number\:of\:vertices$ and $E = number\:of\:edges$ 
- Like BFS, in the worst case, we visit every vertex and every edge.

*Space Complexity*
- $O(V)$ for the visited set and the recursion stack (or stack for iterative DFS).

Applications
- Pathfinding and Cycle Detection
- Topological Sorting
- Solving puzzles
- Component Search
- Maze solving
- Tree traversals
- Finding paths in a graph

Advantages
- Easier to implement
- Works well in situations where you need to explore all possible situations.
- Can be more memory-efficient than BFS in sparse-graphs.

Disadvantages
- May not find the shortest path in unweighted graphs
- DFS can get stuck in infinite loops in cyclic graphs unless a proper cycle detection is in place.
- It may consume a lot of memory for deep recursion (or a large stack for deep graphs)

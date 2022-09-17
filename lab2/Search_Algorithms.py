import collections
from queue import PriorityQueue


class Graph:
    def __init__(self, filename=None):
        # 邻接表
        self._adj = dict()
        # 边数
        self._edge = 0
        # 启发函数
        self._h = dict()

        if filename is not None:
            file = open(filename)
            line = file.readline()
            while line:
                road = line.split()
                if len(road) > 0:
                    self.addEdge(road[0], road[1], road[-1])
                line = file.readline()

    def addEdge(self, v, w, weight):
        if not self.hasVertex(v):
            self._adj[v] = {}
        if not self.hasVertex(w):
            self._adj[w] = {}
        if not self.hasEdge(v, w):
            self._edge += 1
            self._adj[v][w] = int(weight)
            self._adj[w][v] = int(weight)

    # 邻接迭代器
    def adjacentTo(self, v):
        return iter(self._adj[v])

    # 顶点迭代器
    def vertices(self):
        return iter(self._adj)

    def iter_h(self):
        return self._h.items()

    def hasVertex(self, v):
        return v in self._adj

    #
    def hasEdge(self, v, w):
        return w in self._adj[v]

    def countE(self):
        return self._edge

    @staticmethod
    def print_path(method, path, second_path=None):
        if len(path) == 0:
            return
        print(method, end=': ')
        if second_path is not None:
            for i in second_path[len(second_path) - 2::-1]:
                path.append(i)
        for i in range(len(path) - 1):
            print(path[i], end='->')
        print(path[len(path) - 1])

    def __str__(self):
        s = ''
        for v in self.vertices():
            s += v + ': '
            for w in self.adjacentTo(v):
                s += w + ' ' + str(self._adj[v][w]) + ','
            s += '\n'
        return s

    # Uninformed Search
    def BFS(self, start, end):
        path = []
        deque = collections.deque()
        visited = set()
        deque.append(start)
        visited.add(start)
        flag = False
        path.append(start)
        while len(deque) > 0:
            node = deque.popleft()
            for x in self.adjacentTo(node):
                if x not in visited:
                    visited.add(x)
                    deque.append(x)
                    path.append(x)
                    if x == end:
                        flag = True
                        break
            if flag:
                break
        self.print_path('BFS', path)

    def DFS(self, start, end):
        path = []
        queue = []
        visited = set()
        queue.append(start)
        while len(queue) > 0:
            node = queue.pop()
            if node in visited:
                continue
            else:
                path.append(node)
                visited.add(node)
                queue.extend(self._adj[node])
            if node == end:
                break
        self.print_path('DFS', path)

    def DLS_Loop(self, start, end, maxDepth, path, visited):
        if start == end:
            path.append(start)
            return True
        if maxDepth <= 0:
            return False
        if start not in visited:
            path.append(start)
            visited.add(start)
        for current in self.adjacentTo(start):
            if self.DLS_Loop(current, end, maxDepth - 1, path, visited):
                return True
        return False

    def DLS(self, start, end, maxDepth):
        path = []
        visited = set()
        if self.DLS_Loop(start, end, maxDepth, path, visited):
            self.print_path('DLS', path)
        else:
            print("DLS Path No Found.")

    def IDDFS(self, start, end, maxDepth):
        for i in range(maxDepth + 1):
            path = []
            visited = set()
            if self.DLS_Loop(start, end, i, path, visited):
                self.print_path('IDDFS', path)
                return
        print("IDDFS Path No Found.")

    def BDS_Loop(self, queue, visited, path, second_visited=None):
        current = queue.pop()
        if second_visited is not None:
            if current in second_visited:
                return visited
        visited.add(current)
        path.append(current)
        for node in self.adjacentTo(current):
            if node not in visited:
                queue.append(node)
        return visited

    def BDS(self, start, end):
        path_start = []
        path_end = []
        queue_start = []
        visited_start = set()
        queue_end = []
        visited_end = set()
        queue_start.append(start)
        visited_start.add(start)
        queue_end.append(end)
        visited_end.add(end)
        while len(queue_start) > 0 and len(queue_end) > 0:

            self.BDS_Loop(queue_start, visited_start, path_start)
            if len(visited_end & visited_start) > 0:
                break
            self.BDS_Loop(queue_end, visited_end, path_end)
            if len(visited_end & visited_start) > 0:
                break
        self.print_path("BDS", path_start, path_end)

    # Informed Search
    def GDFS(self, start, end, F=False):
        pd_open = PriorityQueue()
        path = []
        set_open = set()
        set_close = set()
        distance = 0
        pd_open.put((0, start))
        set_open.add(start)
        flag = False
        while not pd_open.empty():
            if flag or start == end:
                break
            dis, node = pd_open.get()
            path.append(node+"("+str(dis)+")")
            distance += dis
            set_close.add(node)
            for i in self.adjacentTo(node):
                if i == end:
                    flag = True
                    distance += self._adj[node][i]
                    path.append(end+"("+str(self._adj[node][i])+")")
                    break
                if i not in set_open and i not in set_close:
                    pd_open.put((self._adj[node][i], i))
                    set_open.add(i)
        if F:
            self.print_path("GDFS", path)
            print("Total distance of GDFS:", distance)
        return distance

    def GBFS(self, start, end, flag=False):
        pd_open = PriorityQueue()
        current_node = start
        close_set = set()
        open_set = set()
        open_set.add(start)
        parents = {start: start}
        while current_node != end:
            for n in self.adjacentTo(current_node):
                if n in close_set:
                    continue
                else:
                    parents[n] = current_node
                    if n not in open_set:
                        open_set.add(n)
                        pd_open.put((self._adj[current_node][n], n))
            if len(open_set) == 0:
                break
            # 选择了open_set中边长最短的边
            dis, current_node = pd_open.get()
            open_set.remove(current_node)
            close_set.add(current_node)
        path = []
        n = end
        distance = 0
        parents[start] = start
        while parents[n] != n:
            path.append(n + "(" + str(self._adj[n][parents[n]]) + ")")
            distance += self._adj[n][parents[n]]
            n = parents[n]
        path.append(start + "(0)")
        path.reverse()
        if flag:
            self.print_path("GBFS", path)
            print("Total distance of GBFS:", distance)
        return distance

    def heuristics(self, city, method="GBFS"):
        if method == "GBFS":
            for i in self.vertices():
                self._h[i] = self.GBFS(i, city)
        else:
            for i in self.vertices():
                self._h[i] = self.GDFS(i, city)

    def print_heuristics_table(self):
        for key, value in self.iter_h():
            print(key + ": " + str(value))

    def a_star(self, start, end):
        open_list = set()
        closed_list = set()
        open_list.add(start)
        g = {start: 0}
        parents = {start: start}
        while len(open_list) > 0:
            n = None
            for v in open_list:
                if n is None or g[v] + self._h[v] < g[n] + self._h[n]:
                    n = v
            if n is None:
                print('Path does not exist!')
                return None
            if n == end:
                path = []
                distance = 0
                while parents[n] != n:
                    distance += self._adj[n][parents[n]]
                    path.append(n + "(" + str(self._adj[n][parents[n]]) + ")")
                    n = parents[n]
                path.append(start + "(0)")
                path.reverse()
                self.print_path("A_star", path)
                print("Total distance of A_star:", distance)
                return distance
            for m in self.adjacentTo(n):
                weight = self._adj[n][m]
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        print("No Found")
        return None


if __name__ == '__main__':
    G = Graph('./distances.txt')
    # print(G)
    print("Uninformed Search:")
    G.BFS('Мурманск', 'Одесса')
    G.DFS('Мурманск', 'Одесса')
    G.DLS('Мурманск', 'Одесса', 10)
    G.IDDFS('Мурманск', 'Одесса', 10)
    G.BDS('Мурманск', 'Одесса')
    print("Informed Search:")
    G.GDFS('Мурманск', 'Одесса', True)
    G.GBFS('Мурманск', 'Одесса', True)
    G.heuristics('Одесса')
    #G.print_heuristics_table()
    G.a_star('Мурманск', 'Одесса')


class Node():
    def __init__(self, key):
        self.key = key
        self.children = {}
        
class Trie():
    def __init__(self):
        self.head = Node(None)
        
    def insert(self, strings):
        cur_node = self.head
        
        for string in strings:
            if string not in cur_node.children:
                cur_node.children[string] = Node(string)
            cur_node = cur_node.children[string]
            
    def dfs(self, cur_node, depth):
        for k, v in sorted(cur_node.children.items()):
            for _ in range(depth):
                print('--', end='')
            print(k)
            self.dfs(v, depth+1)
            
            
N = int(input())
        
trie = Trie()
for _ in range(N):
    cur_input = list(map(str, input().split()))
    trie.insert(cur_input[1:])
    
trie.dfs(trie.head, 0)
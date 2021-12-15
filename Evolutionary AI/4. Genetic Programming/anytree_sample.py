from anytree import Node, RenderTree

udo = Node("+")
marc = Node("*", parent=udo)
siv = Node('sqr', parent=marc)
lian = Node("3", parent=siv)
lian = Node("2", parent=marc)
dan = Node("-", parent=udo)
jet = Node("sqr", parent=dan)
din = Node("1", parent=jet)
jan = Node("x", parent=dan)

for pre, fill, node in RenderTree(udo):
    print(f'{pre}{node.name}')

print('marc.ancestors =', marc.ancestors)
print('marc.children =', marc.children)
print('marc.depth =', marc.depth)
print('marc.descendants =', marc.descendants)
print('marc.height =', marc.height)
print('marc.is_leaf =', marc.is_leaf)
print('marc.is_root =', marc.is_root)
print('marc.name =', marc.name)
print('marc.parent =', marc.parent)
print('marc.path =', marc.path)
print('marc.root =', marc.root)
print('marc.siblings =', marc.siblings)

print(len(udo.descendants)+1)
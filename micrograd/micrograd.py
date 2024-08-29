#%%
class Value:
    def __init__(self, data, _children=(),_op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
    def __repr__(self):
        return f"Value=(data={self.data})"

    def __add__(self,value):
        result = Value(self.data + value.data,(self,value),'+')
        return result 
    
    def __mul__(self,value):
        result = Value(self.data * value.data,(self,value), '*')
        return result
    
    def __pow__(self,value):
        if not isinstance(value, Value):
            value = Value(value)
        result = Value(self.data ** value.data, (self,value),'**')
        return result


from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg',graph_attr={'rankdir':'TB'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{data %4f}" % n.data, shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
        
    
    for n1,n2 in edges:
        dot.edge(str(id(n1)),str(id(n2))+ n2._op)
    
    return dot

a = Value(2.0)
b = Value(4.0)
d = Value(-2.0)
c = a + b * d**2
print(c)

print(c._prev)
print(c._op)

draw_dot(c)
# %%

class Value:
    def __init__(self, data, _children=(), _op='', no_grad=False):
        self.data = data 
        self.grad = 0
        self._prev = set(_children) if _children else set()
        self._op = _op
        self._backward = lambda: None
        self.no_grad = no_grad
        self._e = 2.718281828459045
    
    def __call__(self):
        return "Value(data={}, grad={})".format(self.data, self.grad)
    def __repr__(self):
        return self.data.__repr__()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out


    def __sub__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        result = Value(self.data - value.data, (self, value), '-')
        
        if self.no_grad:
            return result
        
        def _backward():
            self.grad += 1 * result.grad
            value.grad += -1 * result.grad
            
        result._backward = _backward
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        result = Value(self.data ** value.data, (self, value), '**')
        return result
    
    def __truediv__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        result = Value(self.data / value.data, (self, value), '/')
        return result
    
    def _exp(self):
        result = Value(self._e ** self.data, (self,))
        return result
        
    def __neg__(self):
        return Value(-self.data, (self,), '-')
    
    def __sqrt__(self):
        return self ** 0.5

    def __iter__(self):
        if isinstance(self.data, list):
            return iter(self.data)
        else:
            return iter([self.data])
    def __lt__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        result = Value(self.data < value.data, (self, value), '<')
        return result

    def __gt__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        result = Value(self.data > value.data, (self, value), '>')
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def relu(self):
        result = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')

        def _backward():
            self.grad += (result.data > 0) * result.grad

        result._backward = _backward
        return result
    
    def tanh(self):
        e_pos = self._exp()
        e_neg = (-self)._exp()
        result = (e_pos - e_neg) / (e_pos + e_neg)
        def _backward():
            self.grad += (1 - result.data ** 2) * result.grad
        result._backward = _backward
        return result
        
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        for node in reversed(topo):
            node._backward()


    @staticmethod
    def from_list(data):
        if isinstance(data, list):
            return [Value(d) for d in data]
        return Value(data)
    
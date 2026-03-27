import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data,dtype=np.float64) if not isinstance(data, np.ndarray) else data
        self._children = _children
        self._op = _op
        self._prev = set(_children)
        self.grad = np.zeros_like(self.data,dtype=np.float64)
        self._backward = lambda: None
    def reduce_grad(grad, shape):
            while len(grad.shape) > len(shape):
             grad = grad.sum(axis=0)
            for i, dim in enumerate(shape):
               if dim == 1:
                  grad = grad.sum(axis=i, keepdims=True)
            return grad 
    def _match_shape(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
             grad = grad.sum(axis=i, keepdims=True)
        return grad       

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += self._match_shape(out.grad, self.data.shape)
            other.grad += self._match_shape(out.grad, other.data.shape)  

        out._backward = _backward
        return out
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')

        def _backward():
            self.grad += self._match_shape(out.grad, self.data.shape)
            other.grad += self._match_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
          grad = out.grad
          print(type(out))
          grad=np.array(grad,dtype=np.float64)
          print("Initial grad shape:", grad.shape)

        # ensure 2D
          if grad.ndim == 0:
            grad = np.array([[grad]])
          print("grad reshaped to 2D:", grad.shape)
          print("self.data shape:", self.data.shape)
          print("other.data shape:", other.data.shape)
          print('self.data transpose shape:',self.data.T.shape)
          print('other.data transpose shape:',other.data.T.shape)

          self.grad += grad @ other.data.T
          other.grad += self.data.T @ grad
          

        out._backward = _backward
        return out
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += self._match_shape(out.grad, self.data.shape)
            other.grad += self._match_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other,(self,),'**')

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad

        out._backward = _backward
        return out
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= Value.reduce_grad(
            (self.data / (other.data ** 2)) * out.grad,
            other.grad.shape
                               )

        out._backward = _backward
        return out
    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out_data = exps / np.sum(exps, axis=-1, keepdims=True)
        out = Value(out_data, (self,), 'Softmax')

        def _backward():
            grad_out = out.grad
            for i in range(len(self.data)):
                for j in range(len(self.data)):
                    if i == j:
                        self.grad[i] += out.data[i] * (1 - out.data[i]) * grad_out[i]
                    else:
                        self.grad[i] -= out.data[i] * out.data[j] * grad_out[j]

        out._backward = _backward
        return out
    def exp(self):
        out_data = np.exp(self.data)
        out = Value(out_data, (self,), 'Exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    def log(self):
        out_data = np.log(self.data)
        out = Value(out_data, (self,), 'Log')

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out
    def layernorm(self, x, gamma, beta, eps=1e-5):
    
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)

        normalized = (x.data - mean) / np.sqrt(var + eps)

        out_data = gamma.data * normalized + beta.data
        out = Value(out_data, (x, gamma, beta), 'LayerNorm')

        def _backward():
          N = x.data.shape[-1]
          grad_out = out.grad

        # gamma, beta gradients
          gamma.grad += np.sum(grad_out * normalized, axis=0)
          beta.grad += np.sum(grad_out, axis=0)

          grad_norm = grad_out * gamma.data

          grad_var = np.sum(
            grad_norm * (x.data - mean) * -0.5 * (var + eps) ** (-1.5),
            axis=-1, keepdims=True
         )

          grad_mean = np.sum(
            grad_norm * -1 / np.sqrt(var + eps),
            axis=-1, keepdims=True
          ) + grad_var * np.mean(-2 * (x.data - mean), axis=-1, keepdims=True)

          x.grad += (
            grad_norm / np.sqrt(var + eps)
            + grad_var * 2 * (x.data - mean) / N
            + grad_mean / N
         )

        out._backward = _backward
        return out
    def crossentropy(self, target):
        assert self.data.shape == target.data.shape
        out_data = -np.sum(target.data * np.log(self.data + 1e-15), axis=-1)
        out = Value(out_data, (self, target), 'CrossEntropy')

        def _backward():
            self.grad += -target.data / (self.data + 1e-15) * out.grad
            target.grad += -np.log(self.data + 1e-15) * out.grad

        out._backward = _backward
        return out
    def dropout(self, p=0.5):
        mask = (np.random.rand(*self.data.shape) > p).astype(float)
        out_data = self.data * mask / (1 - p)
        out = Value(out_data, (self,), 'Dropout')

        def _backward():
            self.grad += (mask / (1 - p)) * out.grad

        out._backward = _backward
        return out
    def sum(self, axis=None, keepdims=False):
        out=Value(np.sum(self.data,axis=axis,keepdims=keepdims),(self,),'Sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    def mean(self, axis=None, keepdims=False):
        out=Value(np.mean(self.data,axis=axis,keepdims=keepdims),(self,),'Mean')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size
        out._backward = _backward
        return out
    def T(self):
        out=Value(self.data.T,(self,),'Transpose')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out
    def reshape(self, *shape):
        out=Value(self.data.reshape(*shape),(self,),'Reshape')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out
    def sqrt(self):
        out=Value(np.sqrt(self.data),(self,),'Sqrt')
        def _backward():
            self.grad += 0.5 / np.sqrt(self.data) * out.grad
        out._backward = _backward
        return out
    def masked_fill(self, mask, value):
        out_data = np.where(mask.data, value, self.data)
        out = Value(out_data, (self, mask), 'MaskedFill')

        def _backward():
            self.grad += np.where(mask.data, 0, out.grad)
            mask.grad += np.where(mask.data, 0, out.grad)

        out._backward = _backward
        return out
    def split(self, num_splits, axis=1):
        splits = np.split(self.data, num_splits, axis=axis)
        out = [Value(s, (self,), 'split') for s in splits]

        def _backward():
          grads = [o.grad for o in out]
          self.grad += np.concatenate(grads, axis=axis)

        for o in out:
          o._backward = _backward

        return out
    def concat(values, axis=1):
        data = np.concatenate([v.data for v in values], axis=axis)
        out = Value(data, values, 'concat')

        def _backward():
          grads = np.split(out.grad, len(values), axis=axis)
          for v, g in zip(values, grads):
            v.grad += g

        out._backward = _backward
        return out
    def reduce_grad(grad, shape):
        while len(grad.shape) > len(shape):
          grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
          if dim == 1:
             grad = grad.sum(axis=i, keepdims=True)
        return grad

    def backward(self):
        topo = []
        visited = set()

        def build_topo(a):
            if a not in visited:
                visited.add(a)
                for child in a._prev:
                    build_topo(child)
                topo.append(a)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
            
# x = Value(np.array([[2.0, 3.0],[4.0, 5.0]]))
# y = Value(np.array([[3.0,9.0],[6.0,8.0]]))
# a=Value(np.array([[1.0,2.0,3.0],[9.0,8.0,7.0]]))
# b=Value(np.array([[2.0,3.0],[4.0,5.0],[6.0,7.0]]))
# z =x+y+x+y+a@b

# z.backward()



# print(a.grad) 
# print(b.grad)
# x = Value(np.array([[2.0, 3.0],
#                     [4.0, 5.0]]))

# z = (x ) ** 2
# z.backward()

# print("x.grad:\n", x.grad)
# x = Value(np.array([[100.0, 200.0, 300.0]]))

# y = x.softmax()


# y.backward()

# print("x.grad:\n", x.grad)
x = Value(np.random.randn(2, 4))

y = x.layernorm()

loss = y.mean()
loss.backward()

print("x.grad:\n", x.grad)
print("gamma.grad:\n", x.gamma.grad)
print("beta.grad:\n", x.beta.grad)


   
 

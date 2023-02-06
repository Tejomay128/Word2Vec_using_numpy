import numpy as np
# from graphviz import Digraph


class Matrix:

    def __init__(self, val, _op='', _desc=()):
        self.val = val
        self.shape = val.shape
        self.grad = np.zeros(self.shape)
        self._backprop = lambda: None
        self._prev = set(_desc)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)
        res = Matrix(self.val + other.val, '+', (self, other))

        def _backprop():
            if self.shape == res.shape:
                self.grad += res.grad
            elif self.shape[0] == res.shape[0]:
                self.grad += res.grad.sum(1, keepdims=True)
            else:
                self.grad += res.grad.sum()
            # print("ADD Self:", self.grad)
            if other.shape == res.shape:
                other.grad += res.grad
            elif other.shape[0] == res.shape[0]:
                other.grad += res.grad.sum(1, keepdims=True)
            else:
                other.grad += res.grad.sum()
            # print("ADD -other:", other.grad)

        res._backprop = _backprop
        return res

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)
        res = Matrix(self.val * other.val, '*', (self, other))

        def _backprop():
            if self.shape == res.shape:
                self.grad += other.val * res.grad
            elif self.shape[0] == res.shape[0]:
                self.grad += (other.val * res.grad).sum(1, keepdims=True)
            else:
                self.grad += (other.val * res.grad).sum()
            # print("Multiply Self:", self.grad)

            if other.shape == res.shape:
                other.grad += self.val * res.grad
            elif other.shape[0] == res.shape[0]:
                other.grad += (self.val * res.grad).sum(1, keepdims=True)
            else:
                other.grad += (self.val * res.grad).sum()
            # print("Multiply -other:", other.grad)

        res._backprop = _backprop
        return res

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * np.array([-1])

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __matmul__(self, other):
        res = Matrix(self.val @ other.val, '@', (self, other))

        def _backprop():
            self.grad += res.grad @ other.val.T
            other.grad += self.val.T @ res.grad

        res._backprop = _backprop
        return res

    def __pow__(self, power):
        res = Matrix(self.val ** power, f'**{power}', (self,))

        def _backprop():
            self.grad += res.grad * (power * (self.val ** (power - 1)))
            # print("Power backprop: ", self.grad)

        res._backprop = _backprop
        return res

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def T(self):
        res = Matrix(self.val.T, 'T', (self,))

        def _backprop():
            self.grad += res.grad.T

        res._backprop = _backprop
        return res

    def log(self):
        res = Matrix(np.log(self.val), _op='log', _desc=(self,))

        def _backprop():
            self.grad += res.grad * (self.val ** -1)

        res._backprop = _backprop
        return res

    def softmax(self):
        # print("softmax:", self.shape)
        shift = self.val - np.max(self.val, axis=1, keepdims=True)
        exp_shift = np.exp(shift)
        smax = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        res = Matrix(smax, 'smax', (self,))

        def _backprop():
            for i in range(smax.shape[0]):
                grad_matrix = -np.outer(smax[i], smax[i]) + np.diag(smax[i])
                self.grad[i] += (res.grad[i].reshape(1, -1) * grad_matrix).sum(1)

        res._backprop = _backprop
        return res

    def cross_entropy(self, gold):
        assert self.shape[0] == gold.shape[0], "number of outputs must be equal"
        argmax_gold = np.argmax(gold.val, axis=1)
        ceLoss = Matrix(np.array((-1.0 / self.shape[0]) * np.sum(np.log(self[np.arange(self.shape[0]), argmax_gold]))),
                        _op='cen',
                        _desc=(self, gold))

        def _backprop():
            self.grad += -1.0 * (gold.val / self.val) * ceLoss.grad
            # no need to backprop for gold values

        ceLoss._backprop = _backprop
        return ceLoss

    def __getitem__(self, index):
        return self.val[index]

    def backprop(self):
        order = []
        visited = set()

        def build_dependency_tree(node):
            if node not in visited:
                visited.add(node)
                for prev_node in node._prev:
                    build_dependency_tree(prev_node)
                order.append(node)
                # print("Step=", v._op)

        build_dependency_tree(self)

        self.grad = np.ones(self.shape)
        for v in reversed(order):
            # print(v.val, "Operator=", v._op, v._backprop)
            v._backprop()
        return order


def zero_grad(order):
    for v in order:
        v.grad = np.zeros(v.grad.shape)


# def trace(root):
#     # builds a set of all nodes and edges in a graph
#     nodes, edges = set(), set()
#
#     def build(v):
#         if v not in nodes:
#             nodes.add(v)
#             for child in v._prev:
#                 edges.add((child, v))
#                 build(child)
#
#     build(root)
#     return nodes, edges
#
#
# def draw_dot(root):
#     dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right
#
#     nodes, edges = trace(root)
#     for n in nodes:
#         uid = str(id(n))
#         # for any value in the graph, create a rectangular ('record') node for it
#         dot.node(name=uid, label=f"Data: {n.val} | Grad: {n.grad}", shape='record')
#         if n._op:
#             # if this value is a result of some operation, create an op node for it
#             dot.node(name=uid + n._op, label=n._op)
#             # and connect this node to it
#             dot.edge(uid + n._op, uid)
#
#     for n1, n2 in edges:
#         # connect n1 to the op node of n2
#         dot.edge(str(id(n1)), str(id(n2)) + n2._op)
#
#     return dot


# a = Matrix(np.array([[1., 2.], [6., 10.], [2., 3.], [3., 5.], [2.5, 3.6]]))
# x = Matrix(np.zeros((2, 2)))
# y = Matrix(np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.]]))

# for epoch in range(10):
#     y_hat = (a @ x).softmax()
#     loss = y_hat.cross_entropy(y)
#     order = loss.backprop()
#     x -= 0.005 * x.grad
#     zero_grad(order)
#     print(f"Epoch {epoch + 1} Loss: {loss.val: .5f}")

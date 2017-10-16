This is just something I was thinking about earlier today. Haven't actually
tested it against other methods yet in terms of speed of convergence or even
if it converges for things harder than simple quadratics.

Simple example:
```python
In [1]: from optimize import optimize
In [2]: f = lambda x, y: x ** 2 + y ** 2
In [3]: optimize(f, 2, start=(-103, 51))
Out[3]: (4.656612873077393e-10, 1.4551915228366852e-11)
```

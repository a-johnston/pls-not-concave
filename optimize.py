""" Some experiment with convex optimization
"""


def _eval_offset(f, X, i, offset):
    temp = list(X)
    temp[i] += offset
    return f(*temp)


def _subgradient(f, X, step=1e-12):
    G = [0] * len(X)
    for i in range(len(X)):
        y1 = _eval_offset(f, X, i, step)
        y2 = _eval_offset(f, X, i, -step)
        G[i] = (y1 - y2) / (2 * step)
    return tuple(G)


def _squared_magnitude(X):
    return sum([x ** 2 for x in X])


def optimize(f, n, start=None, block=10, error=1e-9, gradient=_subgradient):
    """ Assuming f is a convex function taking n real arguments, returns an
    approximation of the minima of f. Optionally takes a given starting point,
    block size, error, and gradient function which otherwise default to the 0
    point of R^n, 10, 1e-3, and a naive subgradient calculation respectively.
    """
    X = list(start or (0,) * n)
    error **= 2
    bounds = [[None, None] for _ in range(n)]
    power = [1 for _ in range(n)]
    while True:
        G = gradient(f, X)
        m = _squared_magnitude(G)

        if m < error:
            break

        g = max(G, key=lambda x: abs(x))
        i = G.index(g)

        if g >= 0:
            bounds[i][1] = X[i]
        if g <= 0:
            bounds[i][0] = X[i]

        temp = 0
        if bounds[i][0] and bounds[i][1]:
            temp = (bounds[i][0] + bounds[i][1]) / 2
        elif bounds[i][0]:
            temp = X[i] + block ** power[i]
            power[i] += 1
        else:
            temp = X[i] - block ** power[i]
            power[i] += 1
        X[i] = temp
    return X


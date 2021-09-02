import scipy

def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

# Comparing arrays
# comparison = array1 == array2
# equal_arrays = comparison.all()
# print(equal_arrays)


def rectangular(module, angle):
    return cmath.rect(module, (angle * math.pi) / 180)


def polar(number):
    return (cmath.polar(number)[0], cmath.polar(number)[1] * (180 / math.pi))


def parallel(z1, z2):
    return (z1 * z2) / (z1 + z2)


def rad(angle):
    return (angle * math.pi) / 180


def deg(angle):
    return (angle * 180) / math.pi

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

A = np.array([[12, 5, 2], [20, 4, 8], [ 2, 4, 3], [ 7, 1, 10]])
print(bmatrix(A) + '\n')


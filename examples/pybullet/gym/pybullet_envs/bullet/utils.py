import numpy as np
import string

def int2base(x, base):
    digs = string.digits + string.ascii_letters
    digits = []

    if x < 0:
        raise ValueError('Got negative number: {}'.format(str(x)))
    elif x == 0:
        return digs[0]

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    digits.reverse()
    return ''.join(digits)

def int2action(x):
    res = int2base(x, 3)
    res += (7 - len(res))*'0'
    return np.array(np.array([int(i) for i in res]))

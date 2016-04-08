import numpy as np

def check(sequence)
    y = True
    flag = True
    set = {}
    for c in sequence:
        if flag:
            if c == '!':
                flag = False
            else:
                set[c]=1
        else:
            y = y and set[c]

    return y



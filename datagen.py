import numpy as np
import string
import random

def check(sequence):
    y = True
    flag = True
    set = {}
    for c in sequence:
        if flag:
            if c == '!':
                flag = False
            else:
                set[c]=True
        else:
            y = y and set.has_key(c)

    return y


def main():
    fx = open('task_progs.txt', 'w')
    fy = open('task_labels.txt', 'w')

    alphabet = list(string.uppercase)
    for _ in xrange(200000):
        a = np.random.random_integers(20,30)
        b = np.random.random_integers(5, 10)
        seen = np.random.choice(alphabet, size=a)
        if np.random.uniform() > 0.5:
            query = np.random.choice(seen, size=b)
        else:
            query = np.random.choice(alphabet, size=b)
        task = np.hstack([seen, '!', query])
        fx.write(''.join(task) + '\n')
        fy.write(str(int(check(task)))+'\n')

    fx.close()
    fy.close()

if __name__ == '__main__':
    main()
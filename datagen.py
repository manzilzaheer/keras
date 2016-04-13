import numpy as np
import string
import random

def check(sequence):
    y = []
    flag = True
    set = {}
    for c in sequence:
        if flag:
            y.append('2')
            if c == '!':
                flag = False
            else:
                set[c]=True
        else:
            y.append(str(int(set.has_key(c))))

    return y


def main():
    fx = open('test3_progs.txt', 'w')
    fy = open('test3_labels.txt', 'w')

    alphabet = list(string.uppercase)
    for _ in xrange(20000):
        a = np.random.random_integers(20,30)
        b = np.random.random_integers(5, 10)
        seen = np.random.choice(alphabet, size=a)
        u = np.random.uniform()
        if  u < 0:
            query = np.random.choice(seen, size=b)
        elif u < 0:
            query = np.random.choice(alphabet,size=b)
        else:
            query = np.random.choice(seen, size=b)
            pos = np.random.randint(0,b,1)
            query[pos] = np.random.choice(np.setdiff1d(alphabet,seen), size=1)
        task = np.hstack([seen, '!', query])
        fx.write(''.join(task) + '\n')
        fy.write(''.join(check(task)) + '\n')
        #fy.write(str(int(check(task)))+'\n')

    fx.close()
    fy.close()

if __name__ == '__main__':
    main()
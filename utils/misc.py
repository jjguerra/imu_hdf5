import os
from output import printout


def filename(sfile=''):

    if os.path.isdir(sfile):
        return sfile.split('/')[-1]
    elif os.path.isfile(sfile):
        return sfile
    else:
        printout(message='Wrong file name/dir passed', verbose=True, extraspaces=1)
        exit(1)


def batch(l, n):
    n = max(1, n)
    r_list = list()
    for index, i in enumerate(xrange(0, len(l), n)):
        r_list.append(l[i:i+n])

    return index, r_list

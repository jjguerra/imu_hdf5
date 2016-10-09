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

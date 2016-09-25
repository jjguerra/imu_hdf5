from datetime import datetime


def printout(message='', verbose=False, time=False, log_file=''):
    if verbose:
        if time:
            print '{0} Time:{1}'.format(message, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            print '{0}'.format(message)

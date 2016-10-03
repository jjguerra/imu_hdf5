from datetime import datetime


def printout(message='', verbose=False, time=False, extraspaces=0, log_file=''):
    if verbose:

        # style format to add extra space after printed message
        if extraspaces != 0 and extraspaces > 0:
            spaces = '\n' * extraspaces
        else:
            spaces = ''

        # provide time
        if time:
            print '{0} Time:{1} {2}'.format(message, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), spaces)

        else:
            print '{0} {1}'.format(message, spaces)

"""
faceswap can put facial features from one face onto another.

Usage: faceswap [options] <image1> <image2>

Options:
    -v --version     show the version.
    -h --help        show usage message.
"""
from docopt import docopt

__version__ = 'demo 1.0'

if __name__ == '__main__':
    arguments = docopt(__doc__, version=__version__)
    print(arguments)

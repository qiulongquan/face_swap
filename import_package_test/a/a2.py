from a import a1
from b import b1
import sys
sys.path.append("..")
import c1 as c11

if __name__ == '__main__':
    a1.aaa()
    b1.bbb()
    c11.ccc()

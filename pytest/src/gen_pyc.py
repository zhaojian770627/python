import imp
import sys

def gen_pyc(name):
    fp,pathname,description=imp.find_module(name)
    try:
        imp.load_module(name, fp, pathname, description)
    finally:
        if fp:
            fp.close()

if __name__=='__main__':
    gen_pyc(sys.argv[1])
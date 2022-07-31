import codecs
import pickle
import random
import sys
import time

while True:
    p = ""
    while True:
        got = input()
        p += got
        if len(p)>0 and p[-1]=="=":
            break

    a,b = pickle.loads(codecs.decode(p.encode(), "base64"))
    time.sleep(1)
    # sys.stderr.write("sending")
    # sys.stderr.write(codecs.encode(pickle.dumps("goodbye"*1000*b), "base64").decode())
    print(codecs.encode(pickle.dumps("goodbye"*1000*b), "base64"))
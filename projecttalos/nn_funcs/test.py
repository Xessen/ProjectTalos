import numpy as np
def hello(se=["relu*2","sigmoid*1"]):
    sequence=[]
    for i in se:
        b,a=i.split("*")[-1],i.split("*")[0]
        for y in range(int(b)):
            sequence.append(a)

    return sequence
hello()

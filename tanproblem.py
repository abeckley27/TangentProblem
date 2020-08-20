#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 01:28:30 2020

@author: aidan
"""

import numpy as np
import primes
from matplotlib import pyplot as plt
import time
import numba

#x = np.linspace(0, 2*np.pi, 100)
#y = np.tan(x)
#plt.plot(x,y)

@numba.jit()
def err(n):
    x = (n+0.5)*np.pi
    return x - int(x)

@numba.jit()
def f(n):
    return int(np.pi * (n + 0.5))


t0 = time.time()
N = 20000000
x = 0
errmin = 1.0
xmin = -1
flag = True

while (x < N):
    errx = err(x)
    if (errx < 0.01):
        ans = f(x)
        if (np.tan(ans) > ans):
            print(ans)
            if (errx < errmin):
                errmin = errx
                xmin = x
        elif (errx < 0.005):
            x += 100
    x += 1

print("------------------")
print(N)
print(xmin)
poop = int((xmin + 0.5) * np.pi)
print(poop)
print(np.tan(poop))
print("------------------")
print("Time: %.2f s" %(time.time() - t0))


def check_plot(M):
    errx = np.array(range(M))
    erry = np.zeros(M)
    for k in errx:
        erry[k] = err(k)
    plt.plot(errx, erry)

def inv_plot(M):
    errx = np.array(range(M))
    erry = np.zeros(M)
    for k in errx:
        erry[k] = err(k)
    plt.plot(errx, 1.0/erry)

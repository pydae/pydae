# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:46:19 2020

@author: jmmau
"""

def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b)
        a, b = b, a + b

    print()
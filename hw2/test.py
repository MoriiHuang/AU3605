import numpy as np
x=[6,4+3j,-3-2j,2-1j,4,2+1j,-3+2j,4-3j]
sum=0
for i in x:
    sum+=abs(i)**2
print(sum,sum/8)
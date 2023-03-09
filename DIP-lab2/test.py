import numpy as np
As=40
Rp=3

print(np.tan(np.pi/8))
print(np.tan(np.pi/4))

s=np.log10((10**(0.1*Rp)-1)/(10**(0.1*As)-1))/(2*np.log10(0.4142/1))
print(s)
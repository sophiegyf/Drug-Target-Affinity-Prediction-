import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

txt_real = r'E:\Yz_Files\CodeWork\CoVAE_3\result\iter0affinities.txt'
txt_pre = r'E:\Yz_Files\CodeWork\CoVAE_3\result\iter0preaffinities.txt'

with open(txt_real,'r') as fi:
    docs = fi.read().splitlines()
    fi.close
# print(docs)
with open(txt_pre,'r') as fo:
    docs_pre = fo.read().splitlines()
    fo.close

d_real=''.join(docs)
d_pre=''.join(docs_pre)
d_real.split()
d_pre.split()
dc_real=','.join(d_real.split())
dc_pre=','.join(d_pre.split())
print(type(dc_real))
x = dc_real.split(',')
x_n=[]
for str in x:
    x_n.append(str[:3])
#x = list(map(int, x))
y = dc_pre.split(',')
y_n=[]
for str in y:
    y_n.append(str[:3])
# print(x)
# print(y)

plt.scatter(x_n,y_n,s=20)
plt.axis([4, 11, 4, 11])
#plt.xlabel('Value', fontsize=14)
#plt.ylabel('predict Value', fontsize=14)
#plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

#定义x、y散点坐标
x = [0,1e-7,5e-7,1e-6,5e-6,1e-5,5e-5,7e-5,7.5e-5,7.55e-5,7.6e-5,8e-5,9e-5,1e-4]
x = np.array(x)
print('x is :\n',x)
num = [6193.14,6191.14,6193.50,6191.51,6196.35,6204.67,6245.74,6263.79,6272.44,6271.49,6273.34,6280.59,6290.45,6297.77]
y = np.array(num)
print('y is :\n',y)
#用3次多项式拟合
f1 = np.polyfit(x, y, 3)
print('f1 is :\n',f1)
p1 = np.poly1d(f1)
print('p1 is :\n',p1)

yvals = p1(x)
print('yvals is :\n',yvals)

std_y = np.ones_like(num) * 6273

#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plot3 = plt.plot(x,std_y,'b',label='6273')
plt.xlabel('mu')
plt.ylabel('R')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('Relation between mu and R')
plt.show()
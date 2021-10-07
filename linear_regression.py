from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import os

#import file
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'V_F.xlsx')
file = pd.read_excel(filename)

x_data = pd.array(file.iloc[:,0])#.reshape(-1, 1)
x = x_data.to_numpy()
y_data = pd.array(file.iloc[:,1])
y = y_data.to_numpy()
n = y.size   #no. of rows

# divide the data in 4 parts
x_a = x[:n//4]
y_a = y[:n//4]
x_b = x[n//4:2*(n//4)]
y_b = y[n//4:2*(n//4)]
x_c = x[2*(n//4):3*(n//4)]
y_c = y[2*(n//4):3*(n//4)]
x_d = x[3*(n//4):]
y_d = y[3*(n//4):]

input = [x_a, x_b, x_c, x_d]
input_parts = [x_a.reshape(-1, 1), x_b.reshape(-1, 1), x_c.reshape(-1, 1), x_d.reshape(-1, 1)]
output_parts = [y_a, y_b, y_c, y_d]

x_con = np.concatenate((x_a, x_b, x_c, x_d))
y_con = np.concatenate((y_a, y_b, y_c, y_d))

#finding the region with least R^2 error
model = LinearRegression()
r_sq = []
intercept = []
for i in range(0,len(input_parts)):
    model.fit(input_parts[i], output_parts[i])
    model = LinearRegression().fit(input_parts[i], output_parts[i])
    error = model.score(input_parts[i], output_parts[i])
    print("r_sq = ", error)
    r_sq.append(error)
    intercept.append(model.intercept_)
    print('intercept:', model.intercept_)

maxi = max(r_sq)
index = r_sq.index(maxi)
print("max = ", maxi, " index = ", index)

model = LinearRegression().fit(input_parts[index], output_parts[index])
output_parts[index] = model.predict(input_parts[index])

x_mean = np.mean(input[index])
y_mean = np.mean(output_parts[index])
  
Sxy = np.sum(x*y)- n*x_mean*y_mean
Sxx = np.sum(x*x)-n*x_mean*x_mean

line_slope = Sxy/Sxx  # Sxy = sample covariance and Sxx = sample variance
line_intercept = y_mean - line_slope*x_mean

print("slope = ", line_slope, " intercept = ", line_intercept)

#predicted values
for i in range(0, len(output_parts)):
    if(i == index):
        continue
    output_parts[i] = (line_slope * input[i]) + line_intercept

y_new = np.concatenate((output_parts[0], output_parts[1], output_parts[2], output_parts[3]))  

# make noisy data from theoretical data
#y_n = y_new + np.random.normal(0, 0.27, len(x_con))

# plot data
#plt.plot(x_con, y_n,"r", label = "noisy data")

plt.plot(x_con, y_con, label="original data")
plt.plot(x_con, y_new, color='yellow', linestyle='dashed', label = "linear data")
plt.legend(loc=0)
plt.show()

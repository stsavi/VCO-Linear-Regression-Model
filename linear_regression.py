from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sympy import * 
import os


#import file
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'V_F.xlsx')
file = pd.read_excel(filename)   #read the file for importing data

x_data = pd.array(file.iloc[:,0])
x = x_data.to_numpy()
y_data = pd.array(file.iloc[:,1])
n = x.size

#finding the closest quadratic curve for given data
curve = np.polyfit(x_data, y_data, 2)   #using polyfit to find the most desirable curve
poly = np.poly1d(curve)    #generate the function using the found coefficients

y = poly(x_data)   #updated data of y

# divide the data in 4 parts
x_a = x[:n//4]
y_a = y[:n//4]
x_b = x[n//4:2*(n//4)]
y_b = y[n//4:2*(n//4)]
x_c = x[2*(n//4):3*(n//4)]
y_c = y[2*(n//4):3*(n//4)]
x_d = x[3*(n//4):]
y_d = y[3*(n//4):]

input_parts = [x_a, x_b, x_c, x_d]   #array of parts of input data
output_parts = [y_a, y_b, y_c, y_d]  #array of parts of output data

x_con = np.concatenate((x_a, x_b, x_c, x_d))  #combined data of x
y_con = np.concatenate((y_a, y_b, y_c, y_d))  #combined data of y

#calculate the second derivative of the parts of curve
second_der = []
for i in range(0, len(input_parts)):
    x1 = symbols('x1')
    
    part_polyfit = np.polyfit(input_parts[i], output_parts[i], 2)   #use polyfit to find the most suitable curve
    new_curve = np.poly1d(part_polyfit)
    new_y = new_curve(input_parts[i])

    expr = part_polyfit[0]*x1**2 + part_polyfit[1]*x1 + part_polyfit[2]

    dydx = Derivative(expr, x1).doit()   #first derivative 
    d2yd2x = Derivative(dydx).doit()   #second derivative

    second_der.append(abs(d2yd2x-0))
    print("second derivative of part", i+1, second_der[i])

mini = min(second_der)
index = second_der.index(mini)
print("min = ", mini, ", index = ", index)

#calculate slope and intercept of the linear region
x_mean = np.mean(input_parts[index])
y_mean = np.mean(output_parts[index])
n = np.size(input_parts[index])

Sxy = np.sum(input_parts[index]*output_parts[index])- n*x_mean*y_mean   #sample covariance
Sxx = np.sum(input_parts[index]*input_parts[index])-n*x_mean*x_mean   #sample variance
  
slope = Sxy/Sxx
intercept = y_mean-slope*x_mean
print("slope =", slope, ", intercept =", intercept)

#updating the values as per the calculated slope and intercept
for i in range(0, len(input_parts)):
    output_parts[i] = slope*input_parts[i] + intercept   #y = mx + c

#concatenated all the parts of curve
output = np.concatenate((output_parts[0], output_parts[1], output_parts[2], output_parts[3]))

#plot the comparision of original and corrected data
plt.plot(x_con, y_con, label="Original data")
plt.plot(x_con, output, label="Corrected data")
plt.legend()
plt.show()

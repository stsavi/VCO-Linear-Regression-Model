import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


file = pd.read_excel("C:\\Users\\Satyam\\Desktop\\VSC\\capstone\\V_F.xlsx")

x = pd.array(file.iloc[:,0]).reshape(-1, 1)
y = pd.array(file.iloc[:,1])

model = LinearRegression()

model.fit(x, y)

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

y_new = model.predict(x)
print(np.square(np.subtract(y_new,y)).mean())
#print(model.)
plt.suptitle('Original vs New curve')
plt.plot(x, y, label = 'Original Data')
plt.plot(x, y_new, label = 'Corrected Data')
plt.legend()
plt.show()

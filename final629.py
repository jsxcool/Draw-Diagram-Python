import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

# read csv file into DataFrame
data = pd.read_csv('Sheet1.csv') 
xTicks = data['Date / Time']
x = np.arange(len(xTicks))
y1 = data['CPU Usage %']
y2 = data['Teperature C']

# (1) time series
figure1 = plt.figure(1)  # create a new figure
plt.xticks(x, xTicks)
plt.plot(x, y1, label='CPU Usage %')
plt.plot(x, y2, label='Teperature C')
plt.title('Time Series')
plt.xlabel('Date / Time')
plt.legend(loc='center')

# (2) Histogram of CPU Usage   number or probability ???????
figure2 = plt.figure(2)
plt.hist(y1, bins=10, density=True)
plt.title('Histogram of CPU Usage')
plt.xlabel('CPU Usage %')
plt.ylabel('probability')

# (3) Histogram of Teperature
figure3 = plt.figure(3)
plt.hist(y2, bins=10, density=True)
plt.title('Histogram of Temperature')
plt.xlabel('Temperature C')
plt.ylabel('probability')

# (4) Horizontal Box Plot of CPU Usage
figure4 = plt.figure(4)
plt.boxplot(y1, vert = False)
plt.title('Horizontal Box Diagram of CPU Usage')
plt.xlabel('CPU Usage %')

# (5) Vertical Box Plot of Temperature
figure5 = plt.figure(5)
plt.boxplot(y2)
plt.title('Vertical Box Diagram of Temperature')
plt.ylabel('Temperature C')

# (6) Scatter: Temperature vs CPU Usage
figure6 = plt.figure(6)
plt.scatter(y1, y2)
plt.title('Scatter Plot')
plt.xlabel('CPU Usage %')
plt.ylabel('Temperature C')
X= np.array(y1).reshape(-1,1)
Y= np.array(y2).reshape(-1,1)
reg = LinearRegression()
reg.fit(X, Y)
slope = reg.coef_
intercept = reg.intercept_
trendline = [slope[0][0]*i+intercept for i in y1]
plt.plot(y1, trendline, color = 'red', lw=2)


# (7) Cross-validation
figure7 = plt.figure(7)
X1 = np.arange(len(X)).reshape(-1,1)  # time series
XX = np.concatenate((X, X1), axis=1)
y_pred = cross_val_predict(reg, XX, Y, cv=5)
plt.scatter(Y, y_pred)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', lw=2)
plt.xlabel('Real Temperature C')
plt.ylabel('Predicted Temperature C')
plt.title('Cross-Validation Predict Temperature')

plt.show()

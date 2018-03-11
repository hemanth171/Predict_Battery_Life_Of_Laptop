import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

with open("battery_trainingdata.txt") as f:
	content = f.readlines()

content = [x.strip() for x in content]
content_split = [s.split(',') for s in content]
col1 = []
col2 = []
for val in content_split:
	col1.append(float(val[0]))
	col2.append(float(val[1]))

df = pd.DataFrame({'Hours_of_Charging': col1,
	'Hours_of_Laptop_Lasted': col2})

hours_below_four = df[df.Hours_of_Charging <= 4.0]

# hours_below_four.plot(kind='scatter', x='Hours_of_Charging', y='Hours_of_Laptop_Lasted')
# plt.show()

battery_hours = hours_below_four['Hours_of_Charging'].values
laptop_hours = hours_below_four['Hours_of_Laptop_Lasted'].values

battery_hours_reshape = hours_below_four['Hours_of_Charging'].values.reshape(-1,1)
laptop_hours_reshape = hours_below_four['Hours_of_Laptop_Lasted'].values.reshape(-1,1)

# battery_hours = df['Hours_of_Charging'].values
# laptop_hours = df['Hours_of_Laptop_Lasted'].values

# battery_hours_reshape = df['Hours_of_Charging'].values.reshape(-1,1)
# laptop_hours_reshape = df['Hours_of_Laptop_Lasted'].values.reshape(-1,1)

lin_model = LinearRegression()
lin_model.fit(battery_hours_reshape, laptop_hours_reshape)

intercept = lin_model.intercept_
coefficient = lin_model.coef_
print(intercept, coefficient)

y_predict = []
for x in battery_hours:
	y_predict.append((x * coefficient) + intercept)

y_predict = np.asarray(y_predict)

X = battery_hours_reshape
Y = y_predict.reshape(-1,1)
plt.scatter(x=battery_hours, y=laptop_hours)
plt.plot(X, Y, linewidth=1.0, color='r')
plt.show()
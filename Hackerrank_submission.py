import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


timeCharged = float(input().strip())
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

battery_hours = hours_below_four['Hours_of_Charging'].values
laptop_hours = hours_below_four['Hours_of_Laptop_Lasted'].values

battery_hours_reshape = hours_below_four['Hours_of_Charging'].values.reshape(-1,1)
laptop_hours_reshape = hours_below_four['Hours_of_Laptop_Lasted'].values.reshape(-1,1)

lin_model = LinearRegression()
lin_model.fit(battery_hours_reshape, laptop_hours_reshape)

intercept = lin_model.intercept_
coefficient = lin_model.coef_

if (timeCharged <= 4.0):
	output = (timeCharged * coefficient) + intercept
	print(output[0][0])
else:
    print('8.00')
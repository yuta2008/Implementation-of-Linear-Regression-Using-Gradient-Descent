# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset containing the city population and corresponding profit values.
2. Initialize the parameters (weight and bias), learning rate, and the number of iterations.
3. Apply the Gradient Descent algorithm to update the parameters by minimizing the cost function.
4. Predict the profit using the trained model and plot the regression line along with the data points.


## Program:
```
import numpy as np

import matplotlib.pyplot as plt

# Sample Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

Y = np.array([1.5, 3.2, 4.1, 5.0, 6.8, 7.9, 8.7, 10.2, 11.0, 12.3], dtype=float)

# Initialize parameters
m = 0          # Slope

b = 0          # Intercept

learning_rate = 0.01

iterations = 1000

n = len(X)

# Gradient Descent
for i in range(iterations):

    Y_pred = m * X + b

    dm = (-2 / n) * np.sum(X * (Y - Y_pred))
    db = (-2 / n) * np.sum(Y - Y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

# Predictions
Y_pred = m * X + b

print("Slope (m):", round(m, 4))

print("Intercept (b):", round(b, 4))

print(f"Regression Equation: Y = {m:.2f}X + {b:.2f}")

# Predict profit for a city with population = 11
population = 11

profit = m * population + b

print(f"Predicted Profit for Population {population}: {profit:.2f}")

# Plot
plt.scatter(X, Y, color='blue', label='Training Data')

plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')

plt.title("Linear Regression using Gradient Descent")

plt.xlabel("Population")

plt.ylabel("Profit")

plt.legend()

plt.grid(True)

plt.show()

```

## Output:
<img width="620" height="521" alt="image" src="https://github.com/user-attachments/assets/69e7f811-1553-4930-bf97-4acd6bb11fec" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(43)

happy_df = pd.read_csv("2018_happiness.csv")
internet_df = pd.read_csv("List of Countries by number of Internet Users - Sheet1.csv")

h_countries = np.array(happy_df["Country or region"])
internet_countries = np.array(internet_df["Country or Area"])

overlap = []
h_scores = []
internet_percents = []
for country in h_countries:
    if country in h_countries and country in internet_countries:
        overlap.append(country)
        h_scores.append(float(happy_df["Score"][h_countries == country]))
        internet_percents.append(float(internet_df["Percentage"][internet_countries == country].iloc[0][:-1]))

ids = np.random.choice(range(len(overlap)), 100, replace=False)

x = np.array(internet_percents).astype("float64")[ids]
y = np.array(h_scores).astype("float64")[ids]  

def generate_LSRL(x, y):
    x_mean = np.mean(x)
    x_std = np.std(x)
    y_mean = np.mean(y)
    y_std = np.std(y)

    x_z = (x - x_mean) / x_std
    y_z = (y - y_mean) / y_std

    r_coef = np.dot(x_z, y_z) / (len(x_z))
    slope = (r_coef * y_std) / x_std
    y_int = y_mean - slope*x_mean

    return r_coef, slope, y_int

def calculate_residuals(x, y, slope, y_int):
    residuals = []
    for i in range(len(x)):
        y_hat = x[i]*slope + y_int
        residuals.append((y[i] - y_hat)**2)

    return residuals

r, slope, y_int = generate_LSRL(x, y)

r2 = r*r

residuals = calculate_residuals(x, y, slope, y_int)
residual = np.sum(residuals)

print("Correlation Coefficient:", r)
print("R^2:", r2)
print("Residual:", residual)
print("Slope:", slope)
print("Y-intercept:", y_int)

plt.figure(figsize=(10, 5))
plt.scatter(x, residuals)
plt.title("Residuals")
plt.xlabel("Internet Access (%)")
plt.ylabel("Residual")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(x, slope*x + y_int, color='orange', label="LSRL")
plt.scatter(x, y)
plt.title("Internet Access vs. Happiness Score")
plt.xlabel("Internet Access (%)")
plt.ylabel("Happiness Score")
plt.legend()
plt.show()
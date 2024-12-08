import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Generate a sample dataset or load your data
data = {
    "YearsExperience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [30000, 35000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000]
}
df = pd.DataFrame(data)

# 2. Split the data into training and testing sets
X = df[["YearsExperience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 6. Save the results to an Excel file
df["PredictedSalary"] = model.predict(X)
df.to_excel("Salary_Prediction.xlsx", index=False)
print("Predictions saved to 'Salary_Prediction.xlsx'.")

# 7. Plot the data and the model's predictions
plt.scatter(df["YearsExperience"], df["Salary"], color="blue", label="Actual Salary")
plt.plot(df["YearsExperience"], df["PredictedSalary"], color="red", label="Predicted Salary")
plt.title("Years of Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

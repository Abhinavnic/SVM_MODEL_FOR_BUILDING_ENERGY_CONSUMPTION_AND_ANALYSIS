import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from google.colab import files

upload=files.upload()

da = pd.read_excel("dataset for buildings energy consumption of 3840 records (1).xlsx")
da

x = da.iloc[:, list(range(0, 10)) + list(range(15, 23))]
x

y = da.iloc[:, 13]
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2529)

svr = SVR(kernel='rbf')
svr.fit(x_train, y_train)

y_pred = svr.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

threshold = y_test.mean()
y_test_binary = (y_test > threshold).astype(int)
y_pred_binary = (y_pred > threshold).astype(int)
accuracy = accuracy_score(y_test_binary, y_pred_binary)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")
print(f"Accuracy: {accuracy:.4f}")


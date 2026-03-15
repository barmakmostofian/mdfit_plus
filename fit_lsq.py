import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score


# Load data
feat_input_file = './merged_data.csv' 
response_input_file = 'exp_pic50.txt' 

df_feat     = pd.read_csv(feat_input_file, sep=',', header=0, index_col=0)
df_response = pd.read_csv(response_input_file, sep=',', header=0, index_col=0)

df_data     = pd.merge(df_feat, df_response, left_index=True, right_index=True)


# Initialize model and LOO strategy
model = LinearRegression()
loo = LeaveOneOut()


# Train and predict using LOO-CV
y_pred = np.zeros(df_data.shape[1])

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred[test_index] = model.predict(X_test)


# Evaluate overall performance
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score:           {r2:.4f}")

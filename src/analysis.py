from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import csv

file_path = 'data/processed/data_binary_clean.csv'
data_binary = pd.read_csv(file_path)

# Split the dataset based on the given year ranges for training and testing
train_data = data_binary[data_binary['year'] < 2566]
test_data = data_binary[data_binary['year'] == 2566]

# Separate features and target variable
train_data_X = train_data.drop(columns=['value'])
train_data_y = train_data['value']

test_data_X = test_data.drop(columns=['value'])
test_data_y = test_data['value']

def Gradient_Boosting_Regression():
    set_random_state_GB = 50
    set_n_estimators_GB = 100
    set_learning_rate_GB = 0.1
    # Initialize and train the XGBoost regressor
    xgb_model = XGBRegressor(random_state=set_random_state_GB, n_estimators=set_n_estimators_GB, learning_rate=set_learning_rate_GB)
    xgb_model.fit(train_data_X, train_data_y)

    # Make predictions
    y_pred_xgb = xgb_model.predict(test_data_X)

    # Evaluate the model
    rmse = mean_squared_error(test_data_y, y_pred_xgb, squared=False)
    mae = mean_absolute_error(test_data_y, y_pred_xgb)
    r2 = r2_score(test_data_y, y_pred_xgb)

    # Display the metrics
    print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

   
    metrics = {
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse, mae, r2]
    }

    out = f"results/figures/Gradient_Boosting_Regression_model_metrics_{set_random_state_GB}_{set_n_estimators_GB}_{set_learning_rate_GB}.csv"
    # File path for the CSV file
    output_file_path = out

    # Write metrics to a CSV file
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(metrics.keys())  # Write header
        writer.writerows(zip(*metrics.values()))  # Write rows

    # Plot "Expected vs Predicted" graph
    # Example: Use the first 10 values for illustration (adjust the range as needed)
    expected_xgb = test_data_y[:20].reset_index(drop=True)  # Actual values
    predicted_xgb = y_pred_xgb[:20]  # Predicted values

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(expected_xgb, label='Expected', marker='o')  # Line plot for Expected values
    plt.plot(predicted_xgb, label='Predicted', marker='o')  # Line plot for Predicted values
    plt.title("Expected vs Predicted Values")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)

    out_g = f"results/output/Gradient_Boosting_Regression_expected_vs_predicted_{set_random_state_GB}_{set_n_estimators_GB}_{set_learning_rate_GB}.png"
    # Save the graph to a file
    graph_file_path = out_g
    plt.savefig(graph_file_path, dpi=300, bbox_inches='tight')

    plt.show()

    print(f"\n Graph saved to {graph_file_path} \n")
    print(f"Metrics saved to {output_file_path} \n")

def Random_Forest():
    set_random_state = 75
    set_n_estimators = 100
    set_max_depth = 10
    # Initialize the Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=set_random_state, n_estimators=set_n_estimators, max_depth=set_max_depth)

    # Train the model on the training data
    rf_model.fit(train_data_X,train_data_y)

    # Make predictions on the test data
    y_pred_rf = rf_model.predict(test_data_X)

    # Evaluate the Random Forest model
    rmse_rf = mean_squared_error(test_data_y, y_pred_rf, squared=False)
    mae_rf = mean_absolute_error(test_data_y, y_pred_rf)
    r2_rf = r2_score(test_data_y, y_pred_rf)

    # rmse_rf, mae_rf, r2_rf

    # Display the metrics
    print(f"RMSE: {rmse_rf}, MAE: {mae_rf}, R2: {r2_rf}")

    metrics = {
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse_rf, mae_rf, r2_rf]
    }

    out = f"results/figures/Random_Forest_model_metrics_{set_random_state}_{set_n_estimators}_{set_max_depth}.csv"
    # File path for the CSV file
    output_file_path = out

    # Write metrics to a CSV file
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(metrics.keys())  # Write header
        writer.writerows(zip(*metrics.values()))  # Write rows

    # Plot "Expected vs Predicted" graph
    # Example: Use the first 10 values for illustration (adjust the range as needed)
    expected_rf = test_data_y[:20].reset_index(drop=True)  # Actual values
    predicted_rf = y_pred_rf[:20]  # Predicted values

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(expected_rf, label='Expected', marker='o')  # Line plot for Expected values
    plt.plot(predicted_rf, label='Predicted', marker='o')  # Line plot for Predicted values
    plt.title("Expected vs Predicted Values")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)

    out_g = f"results/output/Random_Forest_expected_vs_predicted_{set_random_state}_{set_n_estimators}_{set_max_depth}.png"
    # Save the graph to a file
    graph_file_path = out_g
    plt.savefig(graph_file_path, dpi=300, bbox_inches='tight')

    plt.show()

    print(f"\n Graph saved to {graph_file_path} \n")
    print(f"Metrics saved to {output_file_path} \n")

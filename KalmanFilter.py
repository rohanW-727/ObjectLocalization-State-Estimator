import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import statistics
import numpy as np
import random
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def KalmanFilter_Test(single_experiment=False):
    predicted_x = 0.0
    uncertainty_state = 0.1
    process_noise = 0.05
    sensor_noise_std = 0.04
    window_size = 10

    num_samples = 300
    xSimpleKalmanError_data = []
    ySimpleKalmanError_data = []
    xAdaptiveKalmanError_data = []
    yAdaptiveKalmanError_data = []
    x_data = []
    y_data, true_yData, adaptive_yData = [], [], []
    adaptive_x_data = []
    measured_values = []

    for i in range(1, num_samples):
        true_sensor_data = np.sin(0.4 * i) + 1
        sensor_noise = sensor_noise_std * np.random.randn()
        sensor_measurement = true_sensor_data + sensor_noise
        measured_values.append(sensor_measurement)

        # Kalman Filter Prediction step
        predicted_P = uncertainty_state + process_noise
        K = predicted_P / (predicted_P + sensor_noise_std)
        updated_state = predicted_x + K * (sensor_measurement - predicted_x)
        uncertainty_state = (1 - K) * predicted_P
        predicted_x = updated_state

        x_data.append(i)
        y_data.append(predicted_x)
        true_yData.append(true_sensor_data)

        Kalman_RMSEVal = np.sqrt((true_sensor_data - predicted_x) ** 2)
        xSimpleKalmanError_data.append(i)
        ySimpleKalmanError_data.append(Kalman_RMSEVal)

        # Adaptive Kalman Filter logic
        if len(measured_values) >= 4:
            relevant_values = measured_values[-window_size:]
            one_degree_prediction = 2 * relevant_values[-1] - relevant_values[-2]
            two_degree_prediction = (
                3 * relevant_values[-1] - 3 * relevant_values[-2] + relevant_values[-3]
            )
            three_degree_prediction = (
                4 * relevant_values[-1]
                - 6 * relevant_values[-2]
                + 4 * relevant_values[-3]
                - relevant_values[-4]
            )

            errors = [
                abs(sensor_measurement - one_degree_prediction),
                abs(sensor_measurement - two_degree_prediction),
                abs(sensor_measurement - three_degree_prediction),
            ]
            best_error_index = np.argmin(errors)
            raw_predicted_value = [
                one_degree_prediction,
                two_degree_prediction,
                three_degree_prediction,
            ][best_error_index]

            recent_variability = np.std(relevant_values)
            adaptive_process_noise = process_noise + 0.05 * recent_variability
            adaptive_sensor_noise_std = sensor_noise_std + 0.05 * recent_variability

            predicted_P = uncertainty_state + adaptive_process_noise
            K = predicted_P / (predicted_P + adaptive_sensor_noise_std)
            updated_state = raw_predicted_value + K * (
                sensor_measurement - raw_predicted_value
            )
            uncertainty_state = (1 - K) * predicted_P

            adaptive_x_data.append(i)
            adaptive_yData.append(updated_state)

        AdaptiveKalman_RMSEVal = np.sqrt((true_sensor_data - updated_state) ** 2)
        xAdaptiveKalmanError_data.append(i)
        yAdaptiveKalmanError_data.append(AdaptiveKalman_RMSEVal)

    avgKalmanRMSE = sum(ySimpleKalmanError_data) / len(ySimpleKalmanError_data)
    avgAdaptiveRMSE = sum(yAdaptiveKalmanError_data) / len(yAdaptiveKalmanError_data)

    if single_experiment:
        # Plot Kalman vs Adaptive vs True Values
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, label="Kalman Filtered Values", color="blue")
        plt.plot(x_data, true_yData, label="True Values", color="green")
        plt.plot(adaptive_x_data, adaptive_yData, label="Adaptive Kalman Filtered Values", color="orange")
        plt.xlabel("Number of Samples")
        plt.ylabel("Values")
        plt.title("Kalman Filter vs Adaptive Kalman Filter vs True Values")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot RMSE
        plt.figure(figsize=(10, 6))
        plt.plot(
            xSimpleKalmanError_data,
            ySimpleKalmanError_data,
            label="Kalman Filtered RMSE Values",
            color="blue",
        )
        plt.plot(
            xAdaptiveKalmanError_data,
            yAdaptiveKalmanError_data,
            label="Adaptive Kalman Filtered RMSE Values",
            color="green",
        )
        plt.xlabel("Number of Samples")
        plt.ylabel("RMSE Values")
        plt.title("Kalman Filtered RMSE Values vs Adaptive Kalman Filtered RMSE Values")
        plt.legend()
        plt.grid()
        plt.show()
        
      

    return avgKalmanRMSE, avgAdaptiveRMSE


# Run single experiment to plot first two graphs
KalmanFilter_Test(single_experiment=True)

# Run 300 experiments for third graph
num_repeats = 300
simple_rmse = []
adaptive_rmse = []

for _ in range(num_repeats):
    avgKalman_rmse, avgAdaptiveKalman_rmse = KalmanFilter_Test()
    simple_rmse.append(avgKalman_rmse)
    adaptive_rmse.append(avgAdaptiveKalman_rmse)

# Plot RMSE across 300 experiments
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_repeats + 1), simple_rmse, label="Average Kalman Filtered RMSE", color="blue")
plt.plot(
    range(1, num_repeats + 1),
    adaptive_rmse,
    label="Average Adaptive Kalman Filtered RMSE",
    color="green",
)
plt.xlabel("Number of Experiments")
plt.ylabel("Average RMSE Values Over 300 Experiments")
plt.title("Average Kalman Filtered RMSE vs Average Adaptive Kalman Filtered RMSE (300 Experiments)")
plt.legend()
plt.grid()
plt.show()




def KalmanFilter(state_matrix, measurement_matrix, reprojection_error_mm):
    state_covariance_matrix = np.array([[10, 0], [0, 10]]) # Error in calculating calculating the velocity x and y of the process model
    needle_deviation_amount = 10 # The object deviates by 10 mm 
    total_process_noise = reprojection_error_mm + needle_deviation_amount # The total process noise which incorporates deviation and reprojection error from camera calibration
    process_noise_covariance_matrix = np.array([[total_process_noise, 0], [0, total_process_noise]]) 
    sensor_noise_covariance_matrix = np.array([[1, 0], [0, 1]])
    state_transition_matrix = np.array([[1, 0], [0, 1]])
    
    # Predicted state and uncertainty
    predicted_state = state_transition_matrix @ state_matrix 
    uncertainty_P = state_transition_matrix @ state_covariance_matrix @ state_transition_matrix.T + process_noise_covariance_matrix
    
    # Corrected Kalman Gain
    Kalman_gain = uncertainty_P @ np.linalg.inv(uncertainty_P + sensor_noise_covariance_matrix)
    
    # Updated estimate
    initial_estimate = predicted_state + Kalman_gain @ (measurement_matrix - predicted_state)
    #Updating the state covariance matrix
    state_covariance_matrix = (np.eye(state_covariance_matrix.shape[0]) - Kalman_gain) @ uncertainty_P
    
    return initial_estimate


    
def AdaptiveKalmanFilter(measurement_matrix, reprojection_error_mm, MeasuredValues_list):
    if len(MeasuredValues_list) < 5:
        raise ValueError("Insufficient measured values for adaptive filtering")

    # Use the most recent values
    relevant_values = MeasuredValues_list[-5:]

    # Predict velocities (X and Y) using polynomial approximations
    one_degree_polynomial_Xprediction = 2 * relevant_values[-1][0] - relevant_values[-2][0]
    two_degree_polynomial_Xprediction = (
        3 * relevant_values[-1][0] - 3 * relevant_values[-2][0] + relevant_values[-3][0]
    )
    three_degree_polynomial_Xprediction = (
        4 * relevant_values[-1][0]
        - 6 * relevant_values[-2][0]
        + 4 * relevant_values[-3][0]
        - relevant_values[-4][0]
    )

    one_degree_polynomial_Yprediction = 2 * relevant_values[-1][1] - relevant_values[-2][1]
    two_degree_polynomial_Yprediction = (
        3 * relevant_values[-1][1] - 3 * relevant_values[-2][1] + relevant_values[-3][1]
    )
    three_degree_polynomial_Yprediction = (
        4 * relevant_values[-1][1]
        - 6 * relevant_values[-2][1]
        + 4 * relevant_values[-3][1]
        - relevant_values[-4][1]
    )

    # Compute errors
    errorsX = [
        abs(measurement_matrix[0, 0] - one_degree_polynomial_Xprediction),
        abs(measurement_matrix[0, 0] - two_degree_polynomial_Xprediction),
        abs(measurement_matrix[0, 0] - three_degree_polynomial_Xprediction),
    ]
    errorsY = [
        abs(measurement_matrix[1, 1] - one_degree_polynomial_Yprediction),
        abs(measurement_matrix[1, 1] - two_degree_polynomial_Yprediction),
        abs(measurement_matrix[1, 1] - three_degree_polynomial_Yprediction),
    ]

    # Select the best model (smallest error)
    best_x_index = np.argmin(errorsX)
    best_y_index = np.argmin(errorsY)

    raw_predicted_velocity_x = [
        one_degree_polynomial_Xprediction,
        two_degree_polynomial_Xprediction,
        three_degree_polynomial_Xprediction,
    ][best_x_index]

    raw_predicted_velocity_y = [
        one_degree_polynomial_Yprediction,
        two_degree_polynomial_Yprediction,
        three_degree_polynomial_Yprediction,
    ][best_y_index]

    # Calculate variability (recent noise)
    recent_variability = np.std([val[0] for val in relevant_values] + [val[1] for val in relevant_values])
    adaptive_process_noise = reprojection_error_mm + 0.05 * recent_variability
    adaptive_sensor_noise_std = 1 + 0.05 * recent_variability  # Adjusted noise

    # Kalman Filtering
    predicted_P = 10 + adaptive_process_noise
    K = predicted_P / (predicted_P + adaptive_sensor_noise_std)

    updated_velocity_x = raw_predicted_velocity_x + K * (
        measurement_matrix[0, 0] - raw_predicted_velocity_x
    )
    updated_velocity_y = raw_predicted_velocity_y + K * (
        measurement_matrix[1, 1] - raw_predicted_velocity_y
    )

    # Combine updated velocities into a state matrix
    updated_state = np.array([[updated_velocity_x, 0], [0, updated_velocity_y]])

    return updated_state

    
    

    
    
                
    
    
    
    
    
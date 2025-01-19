import numpy as np
import cv2 
import glob
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter
from KalmanFilter import AdaptiveKalmanFilter





cap = cv2.VideoCapture(0) # 0 for the first camera which is my Macbook Pro camera

lower_yellow = np.array([90, 50, 150]) # Lowest range of yellow
upper_yellow = np.array([130, 255, 255]) # Highest range of yellow



# Using Matplotlib for real time plotting
plt.ion()
fig, ax = plt.subplots()
x_data = [] # Time and Velocity data
y_data_measured, y_data_estimated, y_data_adaptive = [], [], []
line_measured, = ax.plot(x_data, y_data_measured, label = "Velocity Measured (mm/s)")
line_estimated, = ax.plot(x_data, y_data_estimated, label = "Velocity Estimated (mm/s)")
line_adaptive, = ax.plot(x_data, y_data_adaptive, label = "Velocity Adaptive (mm/s)")
ax.set_xlim(0, 200) # The limit is 20 seconds
ax.set_ylim(0,1000) # The velocity limit is 1000 mm/s
ax.set_xlabel("Time Elapsed (seconds)") # x label
ax.set_ylabel("Velocity") # y label
ax.set_title("Instantaneous Velocity") # graph title
ax.legend()

frame_count = 0

# Load the checkboard images
checkboard_images = glob.glob('/Users/rohanwadhwa/Documents/Checkboard_Images/*.jpg')

for fname in checkboard_images:
    if(fname is not None):
        print(f"This is working, file name is: {fname}")
    else:
        print("error")

# Initialize the checkboard x and y dimensions
checkboard_dim = (8, 6)

# The 3D coordinate plane based on the number of checkerboard corners which is x * y and 3 due to there being x, y, and z. It is filled with 0s. 
objp = np.zeros((checkboard_dim[0] * checkboard_dim[1], 3), np.float32)

# First creates two arrays, one for x values and then one for y values. After transposition and reshaping it creates multiple 2D arrays of size cols as rows and 2 for cols.
objp[:, :2] = np.mgrid[0:checkboard_dim[0], 0:checkboard_dim[1]].T.reshape(-1,2)

# Scales by grid size which is 15mm x 15mm for each square
objp *= 15

# 3D object points array
obj3DPoints = []

# 2D image points array
img2DPoints = []


if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Creating Background Subtraction using Mixture of Gaussians v2(MOG v2) and ensuring shadows are detected
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Initialize variables to store the previous center of the largest bounding box
prev_center = None
fps = cap.get(cv2.CAP_PROP_FPS) # get the frames per second(fps)
delta_t = 1 / fps # the time interval between each frame is 1 / fps


for fname in checkboard_images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image: {fname}")
        continue  # Skip to the next image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # This finds the checkerboard corners using the gray scale version of the image and the predefined checkerboard dimensions
    ret, corners = cv2.findChessboardCorners(gray, checkboard_dim, None)
      
    # if the corners were found then add the 3D object point arrays to the Obj3Dpoints list and the corners to the img2Dpoints  
    if ret: 
        # Refining the corners to sub-pixel accuracy
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        
        obj3DPoints.append(objp)    
        img2DPoints.append(corners)
    if not ret:
        print(f"Checkerboard corners not found in {fname}")
            

if len(obj3DPoints) > 0 and len(img2DPoints) > 0:
    # Start camera calibration
    reprojection_error, cam_mtx, distortion_coeffs, rotate_vectors, translation_vectors = cv2.calibrateCamera(obj3DPoints, img2DPoints, gray.shape[::-1], None, None)
    
    
    # Extract intrinsic parameters which is the principal point and focal lengths
    focalLength_x = cam_mtx[0, 0]
    focalLength_y = cam_mtx[1, 1]
    # The principal points is the position where the optical axis intersects the 2D image plane
    principalPoint_x = cam_mtx[0, 2] 
    principalPoint_y = cam_mtx[1, 2]
    
    # Compute pixel-to-mm conversion
    checkerboard_square_mm = 15
    square_size_pixels = np.linalg.norm(corners[0] - corners[1])  # Using Euclidian distance formula to determine the length in pixel of each square
    pixel_to_mm = checkerboard_square_mm / square_size_pixels # Getting the number of pixels per millimeter
    



measured_values_list = []
x_data_adaptive = []
while True: # This is an infinite loop that ensures continuous capturing of frames
    ret, frame = cap.read() # Captures each frame continuously
    if not ret: # Frame is not being captured
        break
    #hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Conversion from BGR(Blue, Green, Red) to HSV(Hue = color, Saturation = vividness, Value = intensity(brightness))
    
    # Apply background subtraction on the original frame (BGR)
    fg_mask = bg_subtractor.apply(frame)

    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for yellow color in the HSV frame
    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Combine background subtraction mask and color mask
    combined_mask = cv2.bitwise_and(fg_mask, mask)
    # A kernel size of 5 rows and 5 columns is created for the image processing
    kernel = np.ones((5,5), np.uint8)
    
    # Using the kernel filter to apply Opening morphological operation to erode and then dilate
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    # Using the kernel filter to apply Opening morphological operation to dilate and then erode
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Contours are lines that represent the object's boundaries
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # The third parameter is the chain approximation method
    


    for contour in contours:

        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour) # Getting the dimensions of the contour
            current_center = (x + w // 2, y + h // 2) # Middle of bounding box

            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2) # Creating a bounding box
            cv2.circle(frame, current_center, 5, (255, 0, 0), -1) # Creating a point in the center of the bounding box
                
            if prev_center is not None: # If the previous center position exists(object moved)
                vx = (current_center[0] - prev_center[0]) / delta_t # Getting the instantaneous velocity magnitude in horizontal direction (Change in x) / (time elapsed)
                vy = (current_center[1] - prev_center[1]) / delta_t # Getting the instantaneous velocity magnitude in vertical direction (Change in y) / (time elapsed)
                measurement_matrix = np.array([[vx, 0], [0, vy]])
                state_matrix = np.array([[0,0], [0,0]]) # initial estimate
                
                
                temp_measured_values = measured_values_list.copy()
                temp_measured_values.append(np.array([vx, vy]))
                
                
                
                velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)     

                
                if pixel_to_mm is not None:
                    velocity_mm = velocity_magnitude * pixel_to_mm
                    reprojection_error_mm = reprojection_error * pixel_to_mm
                    acceleration_magnitude = velocity_magnitude / delta_t
                    acceleration_mm = acceleration_magnitude * pixel_to_mm
                    reprojectionError_text = f"Reprojection Error: {reprojection_error:.2f} px"
                else:
                    velocity_text = f"Velocity: {velocity_magnitude:.2f} px/s"
                    cv2.putText(frame, velocity_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
                if len(temp_measured_values) >= 5:
                    x_data_adaptive.append(frame_count)
                    measured_values_array = np.array(temp_measured_values)
                    current_measurement = np.array([[vx, 0], [0, vy]])
                    adaptive_velocity_estimate = AdaptiveKalmanFilter(current_measurement, reprojection_error_mm, measured_values_array)
                    velocity_adaptive_x = adaptive_velocity_estimate[0, 0]
                    velocity_adaptive_y = adaptive_velocity_estimate[1, 1]
                    adaptive_velocity_magnitude = np.sqrt(velocity_adaptive_x**2 + velocity_adaptive_y**2)
                    adaptive_velocity_mm = adaptive_velocity_magnitude * pixel_to_mm
                    y_data_adaptive.append(adaptive_velocity_mm)
                    line_adaptive.set_xdata(x_data_adaptive)
                    line_adaptive.set_ydata(y_data_adaptive)
                
                measured_values_list.append(np.array([vx, vy]))
                if len(measured_values_list) >= 5:
                    measured_values_list.pop(0)
                    
                
                velocity_estimate = KalmanFilter(state_matrix, measurement_matrix, reprojection_error_mm)
                velocity_x = velocity_estimate[0, 0]
                velocity_y = velocity_estimate[1, 1]
                estimated_velocityMagnitude = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
                velocityEstimate_mm = estimated_velocityMagnitude * pixel_to_mm
                
                
                velocity_text = f"Velocity: {velocity_mm:.2f} mm/s"
                cv2.putText(frame, velocity_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
                # Increment frame count and append it as the x data and append the velocity as the y data
                frame_count += 1
                x_data.append(frame_count)
                y_data_measured.append(velocity_mm)
                y_data_estimated.append(velocityEstimate_mm)
                
                
                line_measured.set_xdata(x_data)
                line_estimated.set_xdata(x_data)
                line_measured.set_ydata(y_data_measured)
                line_estimated.set_ydata(y_data_estimated)
                ax.set_xlim(0, max(x_data) + 10)
                ax.set_ylim(0, max(max(y_data_measured, default=0), max(y_data_estimated, default=0), max(y_data_adaptive, default=0)) + 10)
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                
            prev_center = current_center # Update the previous center with the coordinate of the current center
            
        else:
            prev_center = None # Previous center is none meaning object is not moving for that frame
    
    
    
        
    cv2.imshow("Original Frame", frame) 
    
    cv2.waitKey(1)



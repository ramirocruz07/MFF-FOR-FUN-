# import cv2
# import numpy as np
# import argparse

# # Argument parser setup
# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
# count_line_position=550
# # Create video capture object
# cap = cv2.VideoCapture("video.mp4")
# min_width_rect=80
# min_height_rect=80

# def center_handle(x,y,w,h):
#     x1=int(w/2)
#     y1=int(h/2)
#     cx=x+x1
#     cy=y+y1
#     return cx,cy
# offset=6
# detect=[]
# counter=0
# # Check if video capture is successful
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Initialize background subtractor based on chosen algorithm
# if args.algo == 'MOG2':
#     algo = cv2.createBackgroundSubtractorMOG2()
# elif args.algo == 'KNN':
#     algo = cv2.createBackgroundSubtractorKNN()
# else:
#     print("Invalid algorithm specified. Please choose 'MOG2' or 'KNN'.")
#     exit()

# while True:
#     # Read frame-by-frame from the video capture
#     ret, frame1 = cap.read()
    
#     # Check if the frame is valid
#     if not ret:
#         print("Cannot receive frame. Exiting ...")
#         break
    
#     # Convert frame to grayscale and apply Gaussian blur
#     grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grey, (3, 3), 5)
    
#     # Apply background subtraction
#     img_sub = algo.apply(blur)
#     dilat=cv2.dilate(img_sub,np.ones((5,5)))
#     Kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#     dilatdata=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,Kernel)
#     dilatdata=cv2.morphologyEx(dilatdata,cv2.MORPH_CLOSE,Kernel)
#     countershape,hierarchy=cv2.findContours(dilatdata,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
#     cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,241),3)
#     for (i,c) in enumerate(countershape):
#         x,y,w,h=cv2.boundingRect(c)
#         validate_counter=(w>=min_width_rect)and (h>=min_height_rect)
#         if not validate_counter:
#             continue
#         cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
#         center=center_handle(x,y,w,h)
#         detect.append(center)
#         cv2.circle(frame1,center,4,(0,0,255),-1)
#         cv2.putText(frame1,"vehiclecounter:"+str(counter),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,244,0))
#         for (x,y) in detect:
#             if y<(count_line_position+offset) and y>(count_line_position-offset):
#                 counter+=1
#                 cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,255,0),3)
#                 detect.remove((x,y))
#                 print("vehicle counter"+str(counter))
#     cv2.putText(frame1,"vehicle counter:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
            
#     # Display original video and background subtracted image
#     cv2.imshow('Video Original', frame1)
#     cv2.imshow('Background Subtracted', dilatdata)
#     image_copy=frame1.copy()
#     image_copy=cv2.drawContours(image_copy,countershape,-1,(0,255,0),thickness=2)
#     # Exit loop if Enter key (Carriage Return) is pressed
    
#     if cv2.waitKey(1) == 13:
#         break
    

# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Vehicle Detection and Counting')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

# Vehicle detection model
vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')  # Replace with appropriate pre-trained model

# Vehicle types
vehicle_types = ['Car', 'Truck', 'Bus', 'Motorcycle']

# Count line position
count_line_position = 550

# Create video capture object
cap = cv2.VideoCapture("video.mp4")

# Minimum width and height of detected vehicles
min_width_rect = 80
min_height_rect = 80

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

offset = 6
detect = []
counter = {vehicle_type: 0 for vehicle_type in vehicle_types}

# Check if video capture is successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize background subtractor based on chosen algorithm
if args.algo == 'MOG2':
    algo = cv2.createBackgroundSubtractorMOG2()
elif args.algo == 'KNN':
    algo = cv2.createBackgroundSubtractorKNN()
else:
    print("Invalid algorithm specified. Please choose 'MOG2' or 'KNN'.")
    exit()

while True:
    # Read frame-by-frame from the video capture
    ret, frame1 = cap.read()

    # Check if the frame is valid
    if not ret:
        print("Cannot receive frame. Exiting ...")
        break

    # Convert frame to grayscale and apply Gaussian blur
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Apply background subtraction
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatdata = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, Kernel)
    dilatdata = cv2.morphologyEx(dilatdata, cv2.MORPH_CLOSE, Kernel)
    countershape, hierarchy = cv2.findContours(dilatdata, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 241), 3)

    for (i, c) in enumerate(countershape):
        x, y, w, h = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)

        if not validate_counter:
            continue

        # Detect vehicles using pre-trained model
        vehicles = vehicle_cascade.detectMultiScale(frame1[y:y + h, x:x + w], scaleFactor=1.1, minNeighbors=4)

        for (vx, vy, vw, vh) in vehicles:
            vehicle_type = None
            for vehicle in vehicle_types:
                if vw * vh >= 3000:  # Truck or Bus
                    vehicle_type = 'Truck'
                elif vw * vh >= 1500:  # Car
                    vehicle_type = 'Car'
                elif vw * vh >= 500:  # Motorcycle
                    vehicle_type = 'Motorcycle'
                elif vw * vh >= 1000:  # Bus
                    vehicle_type = 'Bus'

            if vehicle_type:
                cv2.rectangle(frame1, (x + vx, y + vy), (x + vx + vw, y + vy + vh), (0, 255, 0), 2)
                center = center_handle(x + vx, y + vy, vw, vh)
                detect.append(center)
                cv2.circle(frame1, center, 4, (0, 0, 255), -1)
                cv2.putText(frame1, f"{vehicle_type} counter: {counter[vehicle_type]}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0))
                counter[vehicle_type] += 1

    for (x, y) in detect:
        if y < (count_line_position + offset) and y > (count_line_position - offset):
            for vehicle_type in vehicle_types:
                counter[vehicle_type] += 1
        detect.remove((x, y))

    # Display vehicle counts
    for vehicle_type, count in counter.items():
        print(f"{vehicle_type} counter: {count}")
        cv2.putText(frame1, f"{vehicle_type} counter: {count}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Display original video and background subtracted image
    cv2.imshow('Video Original', frame1)
    cv2.imshow('Background Subtracted', dilatdata)

    # Exit loop if Enter key (Carriage Return) is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
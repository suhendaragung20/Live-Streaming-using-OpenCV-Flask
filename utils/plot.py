#!/usr/bin/env python

'''
Contain functions to draw Bird Eye View for region of interest(ROI) and draw bounding boxes according to risk factor
for humans in a frame and draw lines between boxes according to risk factor between two humans. 
'''

__title__           = "plot.py"
__Version__         = "1.0"
__copyright__       = "Copyright 2020 , Social Distancing AI"
__license__         = "MIT"
__author__          = "Deepak Birla"
__email__           = "birla.deepak26@gmail.com"
__date__            = "2020/05/29"
__python_version__  = "3.5.2"

# imports
import cv2
import numpy as np

# Function to draw Bird Eye View for region of interest(ROI). Red, Yellow, Green points represents risk to human. 
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk

def filter_inside_roi(frame, bottom_points, boxes, w_roi, h_roi, risk_count):
    h = frame.shape[0]
    w = frame.shape[1]

    scale_w = w_roi / w
    scale_h = h_roi / h    

    hh = h_roi
    ww = w_roi 
 
    data_position_real_center = []
    data_position_real_bottom = []
    data_bird_view = []

    num_inside = 0
    num_outside = 0
    for i, r in zip(bottom_points, boxes):
        pos_x = int(i[0]  * scale_w)
        pos_y = hh - (int(i[1] * scale_h))
        
        if pos_x >= 0 and pos_x <= ww and pos_y >= 0 and pos_y <= hh:

            posision_bird = (pos_x, pos_y)    

            (startX, startY, w, h) = r

            position_real_center = (startX + int(w/2), startY + int(h/2))

            endX = startX + w
            endY = startY + h
            position_real_bottom = (int((endX - startX)/2) + startX, endY)

            data_position_real_center.append(position_real_center)
            data_position_real_bottom.append(position_real_bottom)
            data_bird_view.append(posision_bird)

            num_inside += 1
        else:
            num_outside += 1

    print('num num_inside : ', num_inside)
    print('num num_outside : ', num_outside)

    return data_position_real_center, data_position_real_bottom, data_bird_view, num_inside, num_outside


def plot_bird_view(frame, d_bird, w_roi, h_roi, violate, threshold):
    h = frame.shape[0]
    w = frame.shape[1]

    scale_w = w_roi / w
    scale_h = h_roi / h

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)
    comvis_color = (238, 127, 108)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white

    blank_image = plot_warning_zone(blank_image)

    for (i, pos_bird) in enumerate(d_bird):
        if i in violate:
            blank_image = cv2.circle(blank_image, pos_bird, 5, red, -1)
            blank_image = cv2.putText(blank_image, str(i), pos_bird, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 8)
            blank_image = cv2.putText(blank_image, str(i), pos_bird, cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
            blank_image = cv2.circle(blank_image, pos_bird, int(threshold/2), red, 4)
        else:
            blank_image = cv2.circle(blank_image, pos_bird, 5, comvis_color, -1)
            blank_image = cv2.putText(blank_image, str(i), pos_bird, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 8)
            blank_image = cv2.putText(blank_image, str(i), pos_bird, cv2.FONT_HERSHEY_SIMPLEX, 0.7, comvis_color, 2)
            blank_image = cv2.circle(blank_image, pos_bird, int(threshold/2), comvis_color, 4)
    return blank_image


def plot_warning_zone(bird_image):
    cv2.rectangle(bird_image, (1, 350), (300, 600), (91,195,117), -1)
    cv2.rectangle(bird_image, (1, 350), (60, 600), (68,68,232), -1)
    cv2.rectangle(bird_image, (200, 350), (300, 600), (68,68,232), -1)
    return bird_image


def bird_eye_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white
    warped_pts = []
    r = []
    g = []
    y = []
    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), red, 2)
            
    for i in range(len(distances_mat)):
                
        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])
        
            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), yellow, 2)
            
    for i in range(len(distances_mat)):
        
        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])
    
    for i in bottom_points:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, green, 10)
    for i in y:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
    for i in r:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 10)
        
    #pad = np.full((100,blank_image.shape[1],3), [110, 110, 100], dtype=np.uint8)
    #cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    #cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #blank_image = np.vstack((blank_image,pad))   
        
    return blank_image
    
# Function to draw bounding boxes according to risk factor for humans in a frame and draw lines between
# boxes according to risk factor between two humans.
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk 
def social_distancing_view(frame, distances_mat, boxes, risk_count):
    
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    
    for i in range(len(boxes)):

        x,y,w,h = boxes[i][:]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)
                           
    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        
        if closeness == 1:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),yellow,2)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),yellow,2)
                
            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),yellow, 2) 
            
    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        
        if closeness == 0:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)
                
            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)
            
    pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    frame = np.vstack((frame,pad))
            
    return frame


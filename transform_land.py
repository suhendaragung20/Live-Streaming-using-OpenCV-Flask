# python social_distance_detector.py --input videos/contoh6.mp4 --output output.avi --display 0

# import the necessary packages
from utils.pyimagesearch import social_distancing_config as config
from utils.pyimagesearch.detection import detect_people
from utils import utills
from utils import plot
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import time



class transform_land:

    def __init__(self):
        tic = time.time()
        self.real_roi = self.roi_real()
        # load the COCO class labels our YOLO model was trained on
        # labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
        # self.LABELS = open(labelsPath).read().strip().split("\n")
        # # derive the paths to the YOLO weights and model configuration
        # weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        # configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])


        # # load our YOLO object detector trained on COCO dataset (80 classes)
        # print("[INFO] loading YOLO from disk...")
        # self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # # check if we are going to use GPU
        # if config.USE_GPU:
        #     # set CUDA as the preferable backend and target
        #     print("[INFO] setting preferable backend and target to CUDA...")
        #     self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #     self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


        # # determine only the *output* layer names that we need from YOLO
        # self.ln = self.net.getLayerNames()
        # self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # print('###load_yolov3', time.time() - tic)

    def roi_real(self):
        image = np.zeros((1080, 1920, 3), np.uint8)

        roi_ax = 750
        roi_ay = 340

        roi_bx = 1350
        roi_by = 340

        roi_cx = 1919
        roi_cy = 1050

        roi_dx = 500
        roi_dy = 1050

        roi_w = 300
        roi_h = 600

        tic = time.time()
        roi = [(roi_ax, roi_ay), (roi_bx, roi_by), (roi_cx, roi_cy), (roi_dx, roi_dy)]
        image = fill_box(image, roi, (255,0,0), 0.1)
        # print("fill_box", time.time() - tic)

        boxes = []
        results = []


        # =====================================================
        roi_ax = 500
        roi_ay = 1050

        roi_bx = 660
        roi_by = 600

        roi_cx = 1555
        roi_cy = 600

        roi_dx = 1919
        roi_dy = 1050

        roi_w = 300
        roi_h = 600

        tic = time.time()
        roi_1 = [(roi_ax, roi_ay), (roi_bx, roi_by), (roi_cx, roi_cy), (roi_dx, roi_dy)]
        image = fill_box(image, roi_1, (0,255,0), 0.11)
        # print("fill_box", time.time() - tic)




        # =====================================================
        roi_ax = 500
        roi_ay = 1050

        roi_bx = 660
        roi_by = 600

        roi_cx = 830
        roi_cy = 600

        roi_dx = 800
        roi_dy = 1050

        roi_w = 300
        roi_h = 600

        tic = time.time()
        roi_2 = [(roi_ax, roi_ay), (roi_bx, roi_by), (roi_cx, roi_cy), (roi_dx, roi_dy)]
        image = fill_box(image, roi_2, (0,0,255), 0.25)
        # print("fill_box", time.time() - tic)



        # =====================================================
        roi_ax = 1410
        roi_ay = 1050

        roi_bx = 1230
        roi_by = 600

        roi_cx = 1555
        roi_cy = 600

        roi_dx = 1919
        roi_dy = 1050

        roi_w = 300
        roi_h = 600

        tic = time.time()
        roi_3 = [(roi_ax, roi_ay), (roi_bx, roi_by), (roi_cx, roi_cy), (roi_dx, roi_dy)]
        image = fill_box(image, roi_3, (0,0,255), 0.25)
        # print("fill_box", time.time() - tic)

        return image



    def top_down_transformation(self, frame, roi, color, a_fill):
        H, W = frame.shape[:2]
        src = np.float32(np.array(roi))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        overlay = frame.copy()

        pnts = np.array(roi, np.int32)
        cv2.fillPoly(overlay, [pnts], color)

        alpha = a_fill  # Transparency factor.

        # Following line overlays transparent rectangle over the image
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


        alpha = 0.6
        overlay = frame.copy()

        # pnts = np.array(roi, np.int32)
        cv2.line(overlay, roi[0], roi[1], (255, 70, 70), 2)
        cv2.line(overlay, roi[1], roi[2], (0, 255, 255), 2)
        cv2.line(overlay, roi[2], roi[3], (255, 70, 70), 2)
        cv2.line(overlay, roi[3], roi[0], (0, 255, 255), 2)

        length_scale = min(H, W)
        radius = int(length_scale * (0.02))

        cv2.circle(overlay, roi[0], radius, (70, 70, 70), -1)
        cv2.circle(overlay, roi[1], radius, (70, 70, 70), -1)
        cv2.circle(overlay, roi[2], radius, (70, 70, 70), -1)
        cv2.circle(overlay, roi[3], radius, (70, 70, 70), -1)

        # cv2.putText(overlay, "A", roi[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 8)
        # cv2.putText(overlay, "A", roi[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 2)

        # cv2.putText(overlay, "B", roi[1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 8)
        # cv2.putText(overlay, "B", roi[1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 2)

        # cv2.putText(overlay, "C", roi[2], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 8)
        # cv2.putText(overlay, "C", roi[2], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 2)
        
        # cv2.putText(overlay, "D", roi[3], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 8)
        # cv2.putText(overlay, "D", roi[3], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 2)

        # frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return prespective_transform, frame


    def show_line(self, frame, threshold, roi, color, alpha):
        # resize the frame and then detect people (and only people) in it
        # frame = imutils.resize(frame, width=700)

        # roi_pixel = [(10,10), (400, 10), (500, 300), (100, 300)]
        # print(roi_pixel)
        prespective_transform, frame = self.top_down_transformation(frame, roi, color, alpha)

        return frame


    def to_roi_pixel(self, frame, roi):
        h, w = frame.shape[:2]

        roi_pixel = []
        for r in roi:
            pos_x, pos_y = r

            pos_y = 100 - pos_y

            pos_x_pixel = int(pos_x*w/100)
            pos_y_pixel = int(pos_y*h/100)
            roi_pixel.append((pos_x_pixel, pos_y_pixel))

        return roi_pixel



    def calc_advance(self, frame, threshold, roi, roi_w, roi_h, boxes):
        # frame = cv2.imread('input.jpg')

        # resize the frame and then detect people (and only people) in it

        # roi_pixel = self.to_roi_pixel(frame, roi)
        roi_pixel = roi
        prespective_transform, frame = self.top_down_transformation(frame, roi_pixel, (255,255,255), 0)

        # print(results)

            # boxes.append((startX, startY, w, h))

        # print(boxes)
        print("bboxe", boxes)
        person_points = utills.get_transformed_points(boxes, prespective_transform)
        person_inside = []

        print("person_points", person_points)

        d_center, d_bottom, d_boxes, d_bird, num_inside, num_outside = plot.filter_inside_roi(frame, person_points, boxes, roi_w, roi_h, 0)

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        # if len(boxes) >= 2:
        #     # extract all centroids from the results and compute the
        #     # Euclidean distances between all pairs of the centroids
        #     centroids = np.array(d_bird)
        #     D = dist.cdist(centroids, centroids, metric="euclidean")

        #     # loop over the upper triangular of the distance matrix
        #     for i in range(0, D.shape[0]):
        #         for j in range(i + 1, D.shape[1]):
        #             # check to see if the distance between any two
        #             # centroid pairs is less than the configured number
        #             # of pixels
        #             # if D[i, j] < config.MIN_DISTANCE:
        #             if D[i, j] < threshold:
        #                 # update our violation set with the indexes of
        #                 # the centroid pairs
        #                 violate.add(i)
        #                 violate.add(j)

        idx = 0
        alpha = 0.8
        overlay = frame.copy()
                        # loop over the results
        for pos_center, pos_bottom in zip(d_center, d_bottom):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            color = (238, 127, 108)
            # if the index pair exists within the violation set, then
            # update the color
            if idx in violate:
                color = (0, 0, 255)
            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.line(overlay, pos_center, pos_bottom, color, 1)
            cv2.circle(overlay, pos_center, 5, color, 2)
            cv2.circle(overlay, pos_bottom, 5, (70, 70, 70), -1)
            cv2.putText(overlay, str(idx), pos_center, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 6)
            cv2.putText(overlay, str(idx), pos_center, cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
            person_width = pos_bottom[1] - pos_center[1]
            cv2.ellipse(overlay, pos_bottom, (person_width, int(person_width/3)), 0, 0, 360, (255, 255, 255), 3)
            cv2.ellipse(overlay, pos_bottom, (person_width, int(person_width/3)), 0, 0, 360, color, 2)
            idx += 1

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # draw the total number of social distancing violations on the
        # output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        # print(text)
        # cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        # cv2.imwrite("result.png", frame)

        bird_image = plot.plot_bird_view(frame, d_bird, roi_w, roi_h, violate, threshold)

        print(d_bird)
        print("zzzzzzzzzzzzzzzzzzzzzzzz")

        return frame, bird_image, len(violate), len(d_bird) - len(violate), d_boxes, d_bird


    def calc_simple(self, frame, threshold):
        # frame = cv2.imread('input.jpg')

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, self.net, self.ln,
            personIdx=self.LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    # if D[i, j] < config.MIN_DISTANCE:
                    if D[i, j] < threshold:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        violate.add(i)
                        violate.add(j)


                        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)
            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
                color = (0, 0, 255)
            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        # draw the total number of social distancing violations on the
        # output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        # cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        cv2.imwrite("result.png", frame)

        return frame, len(violate), len(results)

    def plot_region(self,image):

        alpha = 0.2
        image = cv2.addWeighted(self.real_roi, alpha, image, 1 - alpha, 0)

        return image


def fill_box(frame, roi, color, a_fill):
    overlay = frame.copy()

    pnts = np.array(roi, np.int32)
    cv2.fillPoly(overlay, [pnts], color)

    alpha = a_fill  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame



def get_bird(tl, image, boxes):
    

    roi_ax = 750
    roi_ay = 340

    roi_bx = 1350
    roi_by = 340

    roi_cx = 1920
    roi_cy = 1080

    roi_dx = 500
    roi_dy = 1080

    roi_w = 300
    roi_h = 555

    roi = [(roi_ax, roi_ay), (roi_bx, roi_by), (roi_cx, roi_cy), (roi_dx, roi_dy)]

    image, bird_image, num_violate, num_clear, d_boxes, d_bird = tl.calc_advance(image, 20, roi, roi_w, roi_h, boxes)

    return bird_image, d_boxes, d_bird


def filter_zone(d_boxes, d_bird, texts, img_out):
    
    for bird, box, text in zip(d_bird, d_boxes, texts):

        x, y = bird

        if x < 60 and y > 350:
            text = "Looking at the window"
            img_out = plot_object(img_out, box, text, (68,68,232))

        elif x > 200 and y > 350:
            text = "Looking at the plants"
            img_out = plot_object(img_out, box, text, (68,68,232))

        else:
            img_out = plot_object(img_out, box, text)

    return img_out

def plot_object(frame, box, text, fill_color=(238, 127, 108)):
    overlay = frame.copy()

    (H, W) = frame.shape[:2]

    box_border = int(W / 400)

    font_size = 0.5
    (startX, startY, endX, endY) = box

    y = startY - 10 if startY - 10 > 10 else startY + 10

    yBox = y + 5

    # fill_color = (238, 127, 108)

    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  (255, 255, 255), box_border+4)


    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  fill_color, box_border+2)

    font_scale = (0.5*box_border)
    thick = box_border

    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=int(thick))[0]
    # set the text start position

    if ((endX - startX)/2)+startX > W/2:
        text_offset_x = endX - text_width
    else:
        text_offset_x = startX
    text_offset_y = yBox
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(overlay, box_coords[0], box_coords[1], fill_color, cv2.FILLED)

    cv2.putText(overlay, text, (text_offset_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), int(thick))

    alpha = 0.6  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame




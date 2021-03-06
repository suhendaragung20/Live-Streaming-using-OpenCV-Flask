from flask import Flask, render_template, Response
import cv2
import time
# import camera
app = Flask(__name__)

# cam = camera.camera('rtsp://admin:QPPZFE@192.168.100.57:554/H.264_stream')
# cam = cv2.VideoCapture('rtsp://admin:QPPZFE@192.168.100.57:554/H.264_stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)



import jetson.inference
import jetson.utils

import argparse
import sys

import numpy as np

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource("rtsp://admin:QPPZFE@192.168.100.98:554/H.264_stream")
output = jetson.utils.videoOutput("cap.jpg")



def skipFrames(timegap, FPS, cap, CALIBRATION):
   latest = None
   while True :  
      for i in range(int(timegap*FPS/CALIBRATION)) :
        _,latest = cap.read()
        if(not _):
           time.sleep(0.5)#refreshing time
           break
      else:
        break
   return latest


def skipFrames(timegap, FPS, cap, CALIBRATION):
    latest = None
    ret = None
    while True :  
        for i in range(int(timegap*FPS/CALIBRATION)) :
            ret,latest = cap.read()
            if(not ret):
                time.sleep(0.5)#refreshing time
                break
        else:
            break
    return latest, ret


def gen_frames():  # generate frame by frame from camera
    FPS = 60
    CALIBRATION = 1.5
    gap = 0.1
    frame = None
    while True:
        print("================================")
        toc = time.time()
#         time.sleep(0.05)
        # Capture frame-by-frame
        # success, frame = camera.read()  # read the camera frame
        # frame, success = skipFrames(gap, FPS, cam, CALIBRATION)
        # try:
        tic = time.time()
        new_frame = detect()
        print("detection time", time.time() - tic)
#             time.sleep(0.1)
        
        # new_frame = cv2.imread("cap.jpg")
        # new_frame = cv2.resize(new_frame, (640, 480))

        frame = new_frame
        print("success")
        print("read time", time.time() - tic)
#             frame = cv2.imread('tes.PNG')
        # except:
        #     frame = None
        #     print("failll")
        #     pass
        # s = time.time()
        if frame is None:
            continue
        tic = time.time()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        print("add to buffer", time.time() - tic)
        print("total time", time.time() - toc)
        # gap = time.time()-s
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def detect():
    tic = time.time()
    img = input.Capture()

    img_out = convert(img)

    print("capture time", time.time() - tic)
    # detect objects in the image (with overlay)
    tic = time.time()
    detections = net.Detect(img, overlay=opt.overlay)
    print("det time", time.time() - tic)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    print(detections)
    # print the detections
    print("detected {:d} objects in image".format(len(detections)))
    
    tic = time.time()
    
    for detection in detections:
        print(detection)
        print("this is top", detection.Top)
        box = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
        if detection.ClassID == 1:
            text = "Person, " + str(int(detection.Confidence * 100)) + "%"
        else:
            text = "Object"
        if detection.ClassID == 1:
            img_out = plot_object(img_out, box, text)
    print("loop time", time.time() - tic)
    print("#####################################")
    # render the image
    
    tic = time.time()
    # output.Render(img)
    print("jetson img type", type(img))
    print("render time", time.time() - tic)
    return img_out

    # update the title bar
#     output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
#     net.PrintProfilerTimes()

    # # exit on input/output EOS
    # if not input.IsStreaming() or not output.IsStreaming():
    #     break


def convert(img):
    aimg = jetson.utils.cudaToNumpy (img, 1920, 1080, 4)
    #print ("img shape {}".format (aimg1.shape))
    aimg1 = cv2.cvtColor(aimg.astype (np.uint8), cv2.COLOR_RGBA2BGR)

    return aimg1




def plot_object(frame, box, text):
    overlay = frame.copy()

    (H, W) = frame.shape[:2]

    box_border = int(W / 400)

    font_size = 0.6
    (startX, startY, endX, endY) = box

    y = startY - 10 if startY - 10 > 10 else startY + 10

    yBox = y + 5

    fill_color = (238, 127, 108)

    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  (255, 255, 255), box_border+4)


    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  fill_color, box_border+2)

    font_scale = (0.6*box_border)
    thick = box_border

    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=int(thick))[0]
    # set the text start position
    text_offset_x = startX
    text_offset_y = yBox
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(overlay, box_coords[0], box_coords[1], fill_color, cv2.FILLED)

    cv2.putText(overlay, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), int(thick))

    alpha = 0.6  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='192.168.100.104', port=5001, debug=True)

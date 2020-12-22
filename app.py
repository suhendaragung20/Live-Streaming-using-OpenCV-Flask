from flask import Flask, render_template, Response
import cv2
import time
app = Flask(__name__)

camera = cv2.VideoCapture('rtsp://admin:QPPZFE@192.168.100.57:554/H.264_stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


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
    while camera.isOpened():
        # Capture frame-by-frame
        # success, frame = camera.read()  # read the camera frame
        frame, success = skipFrames(gap, FPS, camera, CALIBRATION)
        s = time.time()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        gap = time.time()-s
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
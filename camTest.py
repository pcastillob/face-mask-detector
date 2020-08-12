import numpy as np
import cv2
import datetime
from threading import Thread

# global variables
stop_thread = False             # controls thread execution
img = None                      # stores the image retrieved by the camera


def start_capture_thread(cap):
    global img, stop_thread

    # continuously read fames from the camera
    while True:
        _, img = cap.read()

        if (stop_thread):
            break


def main():
    global img, stop_thread

    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture(0)

    # retrieve properties of the capture object
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_sleep = int(1000 / cap_fps)
    print('* Capture width:', cap_width)
    print('* Capture height:', cap_height)
    print('* Capture FPS:', cap_fps, 'wait time between frames:', fps_sleep)

    # start the capture thread: reads frames from the camera (non-stop) and stores the result in img
    t = Thread(target=start_capture_thread, args=(cap,), daemon=True) # a deamon thread is killed when the application exits
    t.start()

    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0
    cur_fps = 0

    while (True):
        # blocks until the entire frame is read
        frames += 1

        # measure runtime: current_time - last_time
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()

        # compute fps but avoid division by zero
        if (elapsed_time != 0):
            cur_fps = np.around(frames / elapsed_time, 1)

        # TODO: make a copy of the image and process it here if needed
        
        # draw FPS text and display image
        if (img is not None):
            cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("webcam", img)

        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if (key == 27):
            stop_thread = True
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
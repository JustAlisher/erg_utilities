import cv2 as cv
import datetime
import time

from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(use_gpu=True, lang='en',rec_algorithm='CRNN')

Conf_threshold = 0.4
NMS_threshold = 0.4
color = (0, 0, 255)
col = (0, 0, 0)

class_name = []
with open('plate.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet('plate_10000.weights', 'plate.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

first_run = cv.imread("first_run.png")
print("1")
classes, scores, boxes = model.detect(first_run, Conf_threshold, NMS_threshold)
ocr.ocr(first_run, det = False, cls = False)
print("2")

cap = cv.VideoCapture(0)#"rtsp://admin:admin123@192.168.1.10/")
frame_counter = 0
ocr_counter = 0
weight = 700
ocr_text = {}
cv.namedWindow("frame", cv.WINDOW_NORMAL)
loc = [0.86375, 0.90625, 0.9375,0.96875]
rec1 = 0.83375
rec2 = 1
last_num = "None"

cv_bool = True

def num_frequency(dict):
    track = {}

    for key, value in dict.items():
        if value not in track:
            track[value] = 0
        else:
            track[value] += 1
    freq = max(track, key=track.get)

    return [freq, track[freq]]

def recognition(frame):
    plate_roi = []
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        y = [0, 0]
        x = [0, 0]
        y[0], x[0], y[1], x[1] = box
        y = sorted(y)
        x = sorted(x)
        plate_roi.append(frame[x[1]:(x[0] + x[1]), y[1]:(y[0] + y[1])])
    return  [plate_roi, boxes]


while True:
    starting_time = time.time()
    ret, frame = cap.read()

    dt = datetime.datetime.now()
    if ret == False:
        break
    #frame = cv.rotate(frame, cv.ROTATE_180)

    if (cv_bool):
        plate_roi_array, box_array = recognition(frame)
        if (len(plate_roi_array)):
            for plate_roi, box in zip(plate_roi_array, box_array):
                try:
                    if 1.2 > (plate_roi.shape[0]/plate_roi.shape[1]) > 0.7:
                        print(plate_roi.shape[0],int(plate_roi.shape[1]/2))
                        cropped1 = plate_roi[0:int(plate_roi.shape[0]/2), 0:plate_roi.shape[1]]
                        cropped2 = plate_roi[int(plate_roi.shape[0]/2):plate_roi.shape[0], 0:plate_roi.shape[1]]
                        cropped3 = cropped2[0:cropped2.shape[0], 0:int(plate_roi.shape[1]/2)]
                        cropped4 = cropped2[0:cropped2.shape[0], int(plate_roi.shape[1]/2):plate_roi.shape[1]]
                        remake = cv.hconcat([cropped1, cropped4, cropped3])
                        result = ocr.ocr(remake, det=False, cls=False)
                        print(result)
                        cv.imshow("cropped",remake)
                        cv.imshow("plate_roi", plate_roi)
                        result = [[]]
                    else:
                        result = ocr.ocr(plate_roi, det = False, cls = False)
                        print(result)
                        cv.imshow("plate_roi", plate_roi)
                except:
                    result = [[]]
                if len(result[0]):
                    try:
                        if result[0][0][1] >= 0.75:
                            last_num = result[0][0][0].replace(" ", "").upper()
                            #print(last_num)
                            if (len(result[0][0][0].replace(" ", ""))>3):
                                ocr_text[ocr_counter] = result[0][0][0].replace(" ", "").upper()
                                ocr_counter += 1

                    except:
                        print("")
                cv.rectangle(frame, box, color, 3)
                cv.putText(frame, "plate", (box[0], box[1]-10),
                           cv.FONT_HERSHEY_COMPLEX, 0.7, color, 1)
    if len(ocr_text):
        number, frequency = num_frequency(ocr_text)
        if frequency > 10:
            cv_bool = False
            last_num = number
            col = (0, 0, 255)
    endingTime = time.time() - starting_time
    fps = 1/endingTime
    # print(fps)
    cv.rectangle(frame, (10, int(frame.shape[0] * rec1)), (300, int(frame.shape[0] * rec2)), (255, 255, 255), -1)
    cv.putText(frame, f'Last Num: {last_num}', (20, int(frame.shape[0] * loc[0])), cv.FONT_HERSHEY_SIMPLEX, .8, col, 2)
    cv.putText(frame, f'Date: {dt.day}.{dt.month}.{dt.year}', (20, int(frame.shape[0] * loc[1])), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    cv.putText(frame, f'Time: {dt.hour}:{dt.minute}:{dt.second}', (20, int(frame.shape[0] * loc[2])), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    cv.putText(frame, f'Weight: {weight}', (20, int(frame.shape[0] * loc[3])), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)

    cv.putText(frame, f'FPS: {int(fps)}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord("r"):
        cv_bool = True
        last_num = "None"
        col = (0, 0, 0)
        ocr_text = {}
        print("restart")
    if key == ord('f'):
        cap = cv.VideoCapture("rtsp://admin:admin123@192.168.1.10/")
    if key == ord('s'):
        cap = cv.VideoCapture("rtsp://admin:admin123@192.168.1.12/")
cap.release()
cv.destroyAllWindows()

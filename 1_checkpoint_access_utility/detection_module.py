import json
import logging
import os
import re
import traceback
from os.path import exists
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from datetime import datetime
import sys
import site  # runtime dep for PaddleOCR
paddle_path = os.path.join(str(Path.cwd()), "paddle_dist")  # Use PaddleOCR from paddle_dist

if exists(os.path.join(paddle_path, "paddle")) and exists(os.path.join(paddle_path, "paddleocr")):
    sys.path.append(paddle_path)

from paddleocr import PaddleOCR, draw_ocr


class DetectionModule:
    def __init__(self):
        """CV model settings"""
        self.Conf_threshold = 0.3
        self.NMS_threshold = 0.3
        self.blue_color = (255, 0, 0)
        self.black_color = (0, 0, 0)
        self.red_color = (0, 0, 255)
        self._path_plate_names: str = os.path.join('YOLO', 'plate.names')
        self._path_plate_cfg: str = os.path.join('YOLO', 'plate.cfg')
        self._path_plate_weights: str = os.path.join('YOLO', 'plate_10000.weights')
        self._path_first_run_img: str = os.path.join('YOLO', 'first_run.png')
        self.class_name = []
        self.model = None
        self.ocr = None  # Инстанс PaddleOCR
        self.net = None  # Инстанс всей модели распознавания гос.номеров
        self.rec1 = 0.9
        self.rec2 = 1
        self.loc = [0.86375, 0.90625, 0.9375,
                    0.96875]  # these are relative coordinates on the place where we draw white rectangle and write today's date and time
        self.initialize_detection_models()

    def initialize_detection_models(self):
        # initialization of OCR and YOLOv4
        self.ocr = PaddleOCR(use_gpu=True, show_log=False, rec_algorithm='CRNN',
                             rec_model_dir=os.path.join('paddle', 'rec', 'en', 'en_PP-OCRv4_rec_infer'),
                             det_model_dir=os.path.join('paddle', 'det', 'en', 'en_PP-OCRv3_det_infer'),
                             cls_model_dir=os.path.join('paddle', 'cls', 'ch_ppocr_mobile_v2.0_cls_infer'),
                             rec_char_dict_path=os.path.join('paddle', 'en_dict.txt'))

        with open(self._path_plate_names, 'r') as f:
            self.class_name = [cname.strip() for cname in f.readlines()]
        self.net = cv2.dnn.readNet(self._path_plate_weights, self._path_plate_cfg)

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

        first_run = cv2.imread(self._path_first_run_img)

        self.model.detect(first_run, self.Conf_threshold, self.NMS_threshold)
        self.ocr.ocr(first_run, det=False, cls=False)

    def plate_number_recognition(self, frame):
        """ This performs plate detection through YOLO and returns the coordinates of plate on the frame """
        plate_roi_array = []
        classes, scores, boxes = self.model.detect(frame, self.Conf_threshold, self.NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            y = [0, 0]
            x = [0, 0]
            y[0], x[0], y[1], x[1] = box
            y = sorted(y)
            x = sorted(x)
            plate_roi = frame[x[1]:(x[0] + x[1]), y[1]:(y[0] + y[1])]
            if (plate_roi.shape[0] * plate_roi.shape[1] == 0):
                continue
            # cv2.imshow("frame", plate_roi)
            if 1.2 > (plate_roi.shape[0] / plate_roi.shape[1]) > 0.7:  # if the number plate is a square
                cropped1 = plate_roi[0:int(plate_roi.shape[0] / 2), 0:plate_roi.shape[1]]
                cropped2 = plate_roi[int(plate_roi.shape[0] / 2):plate_roi.shape[0], 0:plate_roi.shape[1]]
                cropped3 = cropped2[0:cropped2.shape[0], 0:int(plate_roi.shape[1] / 2)]
                cropped4 = cropped2[0:cropped2.shape[0], int(plate_roi.shape[1] / 2):plate_roi.shape[1]]
                crop_height = [cropped1.shape[0], cropped3.shape[0],
                               cropped4.shape[0]]  # collecting the heights of all cropped parts
                crop_height.sort()  # sorting to have the smallest at the [0] position
                cropped1 = cv2.resize(cropped1, (cropped1.shape[1], crop_height[0]),
                                      interpolation=cv2.INTER_AREA)  # resizing the height of each
                cropped3 = cv2.resize(cropped3, (cropped3.shape[1], crop_height[0]), interpolation=cv2.INTER_AREA)
                cropped4 = cv2.resize(cropped4, (cropped4.shape[1], crop_height[0]), interpolation=cv2.INTER_AREA)
                plate_roi = cv2.hconcat(
                    [cropped1, cropped4, cropped3])  # assemble of a regular number from the cuts of the square one
            plate_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)  # turning image to gray
            plate_roi = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_OTSU)[
                1]  # filtering plate image to get it black-white only
            plate_roi_array.append(plate_roi)

        detected_number = None

        if len(plate_roi_array):
            for plate_roi, box, score in zip(plate_roi_array, boxes, scores):  # drawing box on the image
                try:
                    result = self.ocr.ocr(plate_roi, det=False, cls=False)
                except:
                    result = [[]]
                if len(result[0]):  # if text was detected
                    try:
                        if result[0][0][1] >= 0.75:  # if OCR confidence is 75%
                            detected_number = self.validated_plate_number_or_none(self.clean_string(result[0][0][0]).upper())
                    except:
                        traceback.print_exc()
                cv2.rectangle(frame, box, (0, 0, 255), 3)  # drawing bounding box near the plate
                cv2.putText(frame, "plate", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
        return detected_number

    def validated_plate_number_or_none(self, plate_number):
        """
        final_result = None
        #final_result = self.keep_only_first_three_digits(plate_number)

        status = False

        formats = [
            r'\d{3}[A-Z]{3}\d{2}',
            r'\d{3}[A-Z]{2}\d{2}',
            r'[A-Z]{1}\d{2}[A-Z]{3}',
            r'[A-Z]{1}\d{2}[A-Z]{2}',
            r'[A-Z]{1}\d{2}[A-Z]{3}\d{2}',
            r'[A-Z]{1}\d{2}[A-Z]{3}\d{3}'
        ]

        # Check if the plate number matches any of the formats
        for pattern in formats:
            if re.fullmatch(pattern, plate_number):
                status = True
        if status:
            final_result = plate_number
        else:
            final_result = None

        return final_result
        """
        return self.keep_only_first_three_digits(plate_number)

    def clean_string(self, word):  # leaves only alphanumeric values in the string
        cleaned = ""
        if (not isinstance(word, str)):
            return
        for _ in range(len(word)):
            if word[_].isalnum():
                cleaned += word[_]
        return cleaned

    def keep_only_first_three_digits(self, input_string):
        digits_only = ''.join(char for char in input_string if char.isdigit())
        digits_only = digits_only[0:3]
        return digits_only

    def visualize_data(self, is_weight_stabilized, is_triggered, recognition_status, frame, last_detected_number=None, last_weight_value=None):
        cv2.rectangle(frame, (10, int(frame.shape[0] * self.rec1)), (300, int(frame.shape[0] * self.rec2)),
                      # drawing date and time rectangle
                      (255, 255, 255), -1)
        stabilization_text = ""
        if is_weight_stabilized:
            stabilization_text = "Stab"
        else:
            stabilization_text = "Nestab"



        if last_detected_number is None:
            cv2.putText(frame, f'Gos.nomer:', (20, int(frame.shape[0] * self.loc[2])),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, self.black_color, 2)
        else:
            if recognition_status:
                cv2.putText(frame, f'Gos.nomer: {last_detected_number}', (20, int(frame.shape[0] * self.loc[2])),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, self.red_color, 2)
            else:
                cv2.putText(frame, f'Gos.nomer: {last_detected_number}', (20, int(frame.shape[0] * self.loc[2])),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, self.black_color, 2)

        if last_weight_value is None:
            cv2.putText(frame, f'Ves, kg: ', (20, int(frame.shape[0] * self.loc[3])),
                        cv2.FONT_HERSHEY_SIMPLEX,.8,(0, 0, 0), 2)
        else:
            if is_triggered:
                cv2.putText(frame, f'Ves, kg: {last_weight_value} '
                                   f'{stabilization_text}', (20, int(frame.shape[0] * self.loc[3])),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, self.blue_color, 2)
            else:
                cv2.putText(frame, f'Ves, kg: {last_weight_value} '
                                   f'{stabilization_text}', (20, int(frame.shape[0] * self.loc[3])),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, self.black_color, 2)

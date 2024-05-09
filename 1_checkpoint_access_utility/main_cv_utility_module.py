import base64
import json
import os
import random
import tempfile
import time
import traceback
from os.path import exists
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from datetime import datetime
import sys
from detection_module import DetectionModule
from threading import Thread, Lock
from config import get_data_from_json
import logging

class MainUtility:
    def __init__(self, action_on_detected_plate_number_method, create_photo_method):
        self.detection_module = DetectionModule()

        self.head_mutex = Lock()
        self.detection_mutex = Lock()
        self.turn_mutex = Lock()

        self.testing_mode = get_data_from_json('is_testing_mode')

        self.weight_limit_trigger = get_data_from_json('trigger')

        self.last_weight_value = 0
        self.action_on_detected_plate_number_method = action_on_detected_plate_number_method
        self.create_photo_method = create_photo_method

        self.which_camera_turn = CameraTurn()
        self.last_recognition_status = RecognitionStatus()

        rtsp_first = get_data_from_json('rtsp_first')
        rtsp_second = get_data_from_json('rtsp_second')

        if get_data_from_json('default_camera_number') == 1:
            self.frame_processor_one = FrameProcessor(self, "First", is_default=True, rtsp_url=rtsp_first)
            self.frame_processor_two = FrameProcessor(self, "Second", is_default=False, rtsp_url=rtsp_second)
        else:
            self.frame_processor_one = FrameProcessor(self, "First", is_default=False, rtsp_url=rtsp_first)
            self.frame_processor_two = FrameProcessor(self, "Second", is_default=True, rtsp_url=rtsp_second)
        self.frame_processor_one.start()
        self.frame_processor_two.start()

    def is_plate_detection_ready(self):
        if self.frame_processor_one.thread_is_active and self.frame_processor_one.thread_is_active:
            return True
        else:
            return False


class FrameProcessor(Thread):
    def __init__(self, main_utility, camera_name, is_default=False, rtsp_url=None):
        super().__init__()
        self.main_utility = main_utility
        self.head_mutex = self.main_utility.head_mutex
        self.detection_mutex = self.main_utility.detection_mutex
        self.which_camera_turn = self.main_utility.which_camera_turn
        self.weight_limit_trigger = self.main_utility.weight_limit_trigger
        self.last_recognition_status = self.main_utility.last_recognition_status
        self.turn_mutex = self.main_utility.turn_mutex

        self.action_on_detected_plate_number_method = self.main_utility.action_on_detected_plate_number_method
        self.create_photo_method = self.main_utility.create_photo_method

        self.testing_mode = self.main_utility.testing_mode

        self.camera_name = camera_name
        self.is_default = is_default
        self.rtsp_url = rtsp_url
        self.frame_capture = None
        self.last_frame = None
        self.frame_copy = None
        self.ocr_text = {}
        self.ocr_counter = 0

        self.current_weight = 0
        self.last_weight_values_list = {
            "one": 0,
            "two": 0,
            "three": 0
        }
        self.is_weight_stabilized = False
        self.last_weight_processing_time = time.time()
        self.last_frame_processing_status = False
        self.last_detected_number = None
        self.last_photo_fixation_status = False
        self.last_detection_frames_count = 0
        self.last_wlimtrig_time = time.time()
        self.last_wlimtrig_status = False

        self.thread_is_active = False

        # Initialization of frame capture process
        self.head_mutex.acquire()
        self.initialize_frame_capture()
        self.head_mutex.release()

    def initialize_frame_capture(self):
        if self.rtsp_url:
            self.frame_capture = cv2.VideoCapture(self.rtsp_url)
            logging.info(f"Инициализация айпи камеры")
        else:
            self.frame_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            logging.info(f"Инициализация вебкамеры")

        if self.frame_capture.isOpened():
            self.thread_is_active = True
            logging.info(f"Thread is active")
        else:
            logging.info(f"Cannot open Video Capture stream")
            print('Cannot open Video Capture stream')

    def get_last_weighing_value(self):
        return self.main_utility.last_weight_value

    def run(self):
        while self.thread_is_active:
            self.process_weight_and_video()
            is_triggered = self.is_weight_limit_triggered()
            if self.should_detection_work(is_triggered):
                self.wlim_trigger_time_functionality()
                if self.plate_detection_functionality_recognized_status():
                    logging.info(f"Recognized from {self.camera_name}: {self.last_detected_number}")
                    if not self.last_photo_fixation_status:
                        self.last_photo_fixation_status = True
                        photo_in_base64, photo_name = self.photographic_fixation()
                        self.create_photo_method(photo_abs_path=photo_name)
                        logging.info(f"Photo fixation from {self.camera_name} due to recognition. "
                              f"Name: {photo_name}")
                        print(f"Photo fixation from {self.camera_name} due to recognition. "
                              f"Name: {photo_name}")
                    print(f"Recognized from {self.camera_name}: {self.last_detected_number}")
                    self.action_on_detected_plate_number_method(plate_number=self.last_detected_number)
                elif not self.last_photo_fixation_status and self.last_detection_frames_count > 4:
                    self.last_photo_fixation_status = True
                    photo_in_base64, photo_name = self.photographic_fixation()
                    self.create_photo_method(photo_abs_path=photo_name)
                    logging.info(f"Photo fixation from {self.camera_name} due to frame count"
                          f"Name: {photo_name}")
                    print(f"Photo fixation from {self.camera_name} due to frame count"
                          f"Name: {photo_name}")
                elif (not self.last_photo_fixation_status and self.is_default and
                      (time.time() - self.last_wlimtrig_time) > 7 and self.is_weight_stabilized):
                    self.last_photo_fixation_status = True
                    photo_in_base64, photo_name = self.photographic_fixation()
                    self.create_photo_method(photo_abs_path=photo_name)
                    logging.info(f"Photo fixation from {self.camera_name} due to stabilization"
                          f"Name: {photo_name}")
                    print(f"Photo fixation from {self.camera_name} due to stabilization"
                          f"Name: {photo_name}")

            self.main_utility.detection_module.visualize_data(self.is_weight_stabilized, is_triggered,
                                                              self.last_recognition_status.status,
                                                              self.last_frame,
                                                              self.last_detected_number, self.current_weight)

            if self.testing_mode:
                self.display_last_frame()

        self.release_video_capture()

    def save_photo(self):
        logging.info("Save photo start")
        photo_name = self.generate_name_for_photo()
        logging.info(f"Generated name for photo: {photo_name}")
        try:
            cv2.imwrite(photo_name, self.frame_copy)
            logging.info(f"Success")
        except Exception as e:
            logging.info(f"Error while saving photo {e}")
            pass

        return photo_name

    def photographic_fixation(self):
        logging.info(f"Photo fixation start")
        _, png = cv2.imencode('.png', self.frame_copy)

        # Convert PNG image to base64 string
        #base64_png = base64.b64encode(png).decode('utf-8')
        base64_png = None

        photo_name = None

        # if self.testing_mode:
        photo_name = self.save_photo()

        return base64_png, photo_name

    def generate_name_for_photo(self):  # returns the name of the file to be saved
        daytime = datetime.now()

        images_folder = os.path.join(str(Path.cwd()), "images")
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)

        name = str(daytime.year) + str(daytime.month) + str(daytime.day) + "_" + str(
            daytime.hour) \
               + str(daytime.minute) + str(daytime.second) + "_" + str(self.camera_name) + '.png'
        name = os.path.join(tempfile.gettempdir(), name)
        return name

    def process_weight_and_video(self):
        self.current_weight = self.get_last_weighing_value()
        at_the_moment = time.time()
        time_difference = at_the_moment - self.last_weight_processing_time
        if time_difference > 1.0:
            # print(f"{self.camera_name}: Current weight - {self.current_weight}, diff - {time_difference}")
            self.last_weight_values_list["three"] = self.last_weight_values_list["two"]
            self.last_weight_values_list["two"] = self.last_weight_values_list["one"]
            self.last_weight_values_list["one"] = self.current_weight
            self.weight_stabilization_check()
            self.last_weight_processing_time = at_the_moment

        self.last_frame_processing_status, self.last_frame = self.frame_capture.read()
        self.frame_copy = self.last_frame.copy()

    def weight_stabilization_check(self):
        if self.last_weight_values_list["three"] > 0 \
                and self.last_weight_values_list["two"] > 0 \
                and self.last_weight_values_list["one"] > 0:
            first_ratio = self.last_weight_values_list["three"] / self.last_weight_values_list["two"]
            second_ratio = self.last_weight_values_list["two"] / self.last_weight_values_list["one"]

            if (0.94 < first_ratio < 1.06) and (0.94 < second_ratio < 1.06):
                self.is_weight_stabilized = True
            else:
                self.is_weight_stabilized = False
        elif self.last_weight_values_list["three"] == 0 \
                and self.last_weight_values_list["two"] == 0 \
                and self.last_weight_values_list["one"] == 0:
            self.is_weight_stabilized = True
        else:
            self.is_weight_stabilized = False

    def is_weight_limit_triggered(self):
        return self.current_weight >= self.weight_limit_trigger

    def wlim_trigger_time_functionality(self):
        if not self.last_wlimtrig_status and self.is_default:
            self.last_wlimtrig_time = time.time()
            self.last_wlimtrig_status = True

    def should_detection_work(self, is_triggered):
        if not self.frame_validated():
            return False

        self.turn_mutex.acquire()
        should_or_not = False
        if is_triggered and self.current_weight != -1:
            if (not self.last_recognition_status.status
                    and (self.which_camera_turn.turn == -1 or self.which_camera_turn.turn == self.camera_name)):
                self.which_camera_turn.turn = self.camera_name
                should_or_not = True
        else:
            self.clear_last_detection_data()
        self.turn_mutex.release()
        return should_or_not

    def is_plate_number_recognized(self):
        is_recognized = False
        try:
            if len(self.ocr_text):  # if recognised text is no empty
                number, frequency = the_most_frequent_plate_number(
                    self.ocr_text)  # looking for the most frequent result
                if frequency > 2:  # if it appears more than 10 times, save the image and turn off computer vision
                    is_recognized = True
        except:
            traceback.print_exc()
        return is_recognized

    def clear_last_detection_data(self):
        self.which_camera_turn.turn = -1
        self.last_recognition_status.status = False
        self.last_detected_number = None
        self.ocr_text = {}
        self.ocr_counter = 0
        self.last_photo_fixation_status = False
        self.last_detection_frames_count = 0
        self.last_wlimtrig_status = False

    def clear_last_detected_photo_with_location(self):
        self.last_detected_number = None

    def plate_detection_functionality_recognized_status(self):
        self.detection_mutex.acquire()
        is_recognized = False
        try:
            self.last_detected_number = self.main_utility.detection_module.plate_number_recognition(
                self.last_frame)
            if self.last_detected_number:
                self.ocr_text[self.ocr_counter] = self.last_detected_number
                self.ocr_counter += 1
                self.last_detection_frames_count += 1
                is_recognized = self.is_plate_number_recognized()
        except:
            traceback.print_exc()

        if is_recognized:
            self.last_recognition_status.status = True
        self.detection_mutex.release()
        return is_recognized

    def display_last_frame(self):
        if self.last_frame is not None:
            cv2.imshow(self.camera_name, self.last_frame)

        c = cv2.waitKey(1)
        if c == 27:
            return

    def release_video_capture(self):
        self.head_mutex.acquire()
        self.frame_capture.release()
        self.head_mutex.release()

    def frame_validated(self):
        return self.last_frame_processing_status and isinstance(self.last_frame, np.ndarray)


def the_most_frequent_plate_number(dict):  # finds the most frequent number read by OCR
    track = {}

    for key, value in dict.items():
        if value not in track:
            track[value] = 0
        else:
            track[value] += 1
    freq = max(track, key=track.get)

    return [freq, track[freq]]


class CameraTurn:
    def __init__(self):
        self.turn = -1


class RecognitionStatus:
    def __init__(self):
        self.status = False

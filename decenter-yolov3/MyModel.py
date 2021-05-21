from decenter.ai.baseclass import BaseClass
from decenter.ai.appconfig import AppConfig
from decenter.ai.requesthandler import AIReqHandler
from decenter.ai.flask import init_handler
from base64 import b64encode
import numpy as np

import logging, sys
import time
import asyncio
from threading import Thread
import os

import cv2
import json
from uuid import uuid4


import paho.mqtt.client as paho
from utils.utils import *
import tensorflow as tf
import pytesseract


class MyModel:
    def __init__(self, app):
        """
        """
        self.app = app
        self.sess = tf.compat.v1.Session()

        self.fps = -1
        self.current_frame_time = -1

        self.source = None
        self.cap = None
        self.thread = None

        self.skip_frames = False
        self.continue_running = False
        self.id = str(uuid4())
        self.model = None

        logging.info('creating session')

    def load_ai_model(self, filename):

        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 0:
            print(f'GPUs {gpus}')
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError:
                pass

        self.CLASSES = "model_data/names.txt"
        self.model = Create_Yolo(input_size=416, CLASSES=self.CLASSES)
        self.model.run_eagerly = True
        self.model.load_weights(f"./model_data/yolov3_custom")

        return 0

    def ai_thread(self, *args):
        def on_connect(client, userdata, flags, rc):
            logging.info("CONNECT RECEIVED with code %d." % (rc))

        def on_publish(client, userdata, mid):
            logging.info("PUBLISHED")

        def on_message(client, userdata, message):
            logging.info("message received ", str(message.payload.decode("utf-8")))
            logging.info("message topic=", message.topic)
            logging.info("message qos=", message.qos)
            logging.info("message retain flag=", message.retain)

        # pytesseract.pytesseract.tesseract_cmd = r'/tesseract-ocr-setup-3.02.02.exe'
        client = paho.Client(transport="websockets")
        client.on_connect = on_connect
        client.on_publish = on_publish
        # client.on_message = on_message
        #logging.info(self.app.appconfig.get_destination()["detected_cor"].hostname)
        #logging.info(self.app.appconfig.get_destination()["detected_cor"].port)
        logging.info(type(self.app.appconfig.get_destination()["detected_cor"].hostname))
        logging.info(type(self.app.appconfig.get_destination()["detected_cor"].port))
        result = client.connect("194.249.2.112", self.app.appconfig.get_destination()["detected_cor"].port)
        client.loop_start()

        # if you need to check the mqtt data send to prometheus you need to uncomment this an on_message line
        # client.subscribe("prometheus/job/AI_metrics/instance/yolov3/monitoring_fps")
        # client.subscribe("prometheus/job/AI_metrics/instance/yolov3/monitoring_video_delay")
        logging.warn('opening source at: ' + self.source)

        # AXIS ip camera MJPEG
        self.cap = cv2.VideoCapture(self.source)
        self.set_minimum_fps(args[0])
        self.current_frame_time = time.time()
        prometheus_time = time.time()
        self.init_video = False


        while self.continue_running:
            # skip frames until the delay is smaller than the value specified
            if self.skip_frames:
                self.current_frame_time += 1 / self.cap.get(cv2.CAP_PROP_FPS)

                if self.get_current_video_delay() < 0.2:
                    self.skip_frames = False
                    self.current_frame_time = time.time()
            else:
                try:
                    ret, frame = self.cap.read()

                    start_time = time.time()  # start time of the loop

                    if ret:
                        height_ori, width_ori = frame.shape[:2]
                        #frame = cv2.resize(frame_ori, tuple([416, 416]))
                        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #frame = np.asarray(frame, np.float32)
                        #frame = frame[np.newaxis, :] / 255.

                        try:
                            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                        except:
                            continue

                        image_data = image_preprocess(np.copy(original_frame), [416, 416])
                        image_data = image_data[np.newaxis, ...].astype(np.float32)

                        pred_bbox = self.model.predict(image_data)

                        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                        pred_bbox = tf.concat(pred_bbox, axis=0)


                        bboxes = postprocess_boxes(pred_bbox, original_frame, 416, 0.4)
                        bboxes = nms(bboxes, 0.5, method='nms')

                        detections = []

                        for box in bboxes:
                            logging.info(box)
                            detections.append({'class': 'license_plate',
                                               'location': [str(box[0]*width_ori/float(416)), str(box[1]*height_ori/float(416)),
                                                            str(box[2]*width_ori/float(416)), str(box[3]*height_ori/float(416))],
                                               'score': str(box[4])})
                            coor = np.array(box[:4], dtype=np.int32)
                            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                            height = int((coor[3] - coor[1]) / 7)
                            gray = gray[coor[1] + height:coor[3] - height, coor[0]:coor[2]]
                            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                            detections[-1]['text'] = pytesseract.image_to_string(gray)

                        frame = draw_bbox(original_frame, bboxes, CLASSES=self.CLASSES, rectangle_colors='')


                        retval, buffer = cv2.imencode('.jpg', frame)
                        jpg_as_text = b64encode(buffer)

                        #infot = client.publish("miha/test_yolo_frame", jpg_as_text, qos=0)
                        #infot.wait_for_publish()


                        message = {'ai_id': self.id,
                                   'fps': self.fps,
                                   'delay': self.get_current_video_delay(),
                                   'timestamp': str(self.current_frame_time),
                                   'detections': detections,
                                   'encoded_image': jpg_as_text}
                        logging.info(message)
                        message = json.dumps(message)

                        """
                        for i in range(len(boxes_)):
                            x0, y0, x1, y1 = boxes_[i]
                            plot_one_box(frame_ori, [x0, y0, x1, y1], label=self.classes[labels_[i]], color=self.color_table[labels_[i]])
                        
                        frame_out = cv2.imencode('.jpg', frame_ori)[1].tostring()
                        """

                        infot = client.publish("miha/test_yolo", message, qos=0)
                        infot.wait_for_publish()

                        # send data to prometheus every 2 seconds
                        if time.time() - prometheus_time > 1:
                            logging.info("publishing to prometeus current FPS")
                            infot = client.publish("prometheus/job/AI_metrics/instance/yolov3/monitoring_fps", self.get_current_fps())
                            infot.wait_for_publish()
                            infot = client.publish("prometheus/job/AI_metrics/instance/yolov3/monitoring_video_delay", self.get_current_video_delay())
                            infot.wait_for_publish()
                            prometheus_time = time.time()

                        # yield(b'--frame\r\n'
                        #   b'Content-Type: image/jpeg\r\n\r\n' + frame_out + b'\r\n')


                        #cv2.imshow('output', frame)


                        self.fps = 1.0 / (time.time() - start_time)
                        self.reset_video_feed()
                        if self.cap.isOpened():
                            self.current_frame_time += 1 / self.cap.get(cv2.CAP_PROP_FPS)
                    else:
                        logging.info("NO DATA!")
                        break
                except Exception as ex:
                    logging.info(f"An exception occured while processing a frame: {type(ex)}: {ex}")

        client.disconnect()
        self.cap.release()
        return "DONE"

    def get_current_fps(self):
        if not self.continue_running:
            return -1
        return self.fps

    def get_current_video_delay(self):
        if not self.continue_running:
            return -1
        return time.time() - self.current_frame_time

    def set_minimum_fps(self, minimum_fps):
        if self.continue_running and 60 >= minimum_fps != self.cap.get(cv2.CAP_PROP_FPS) and minimum_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, minimum_fps)
            return True
        return False

    def reset_video_feed(self):
        if self.get_current_video_delay() > 1:
            print("Delay was grater then the limit, so video feed was reset to align with the current stream")
            self.skip_frames = True

    def start_thread(self, fps):
        if not self.continue_running:
            print("starting AI computations")
            self.thread = Thread(target=self.ai_thread, args=[fps])
            self.continue_running = True
            self.thread.start()
            return "STARTED"
        return "ALREADY RUNNING"

    def stop_thread(self):
        print("Stopping AI computations")
        if self.continue_running:
            self.continue_running = False
            self.fps = -1
            self.current_frame_time = - 1
            self.cap.release()
            return "AI STOPPED"
        return "AI WAS NOT RUNNING"

    def compute_ai(self, *args):
        self.source = ''.join(args[0])
        # if we would need auto start of the AI model
        self.start_thread(30)
        return {}

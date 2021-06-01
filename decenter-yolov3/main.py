from decenter.ai.baseclass import BaseClass
from decenter.ai.appconfig import AppConfig
from decenter.ai.requesthandler import AIReqHandler
from decenter.ai.flask import init_handler
import decenter.ai.utils.model_utils as model_utils
import paho.mqtt.client as paho


import logging
import sys
import os
import json
from flask import request

from MyModel import MyModel
from utils.train import train

TRAINING = False


def main():

    # set logger config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    # Init BaseClass
    if os.getenv('MY_APP_CONFIG') is None:
        app = BaseClass()
    else:
        app = BaseClass(json.loads(os.getenv('MY_APP_CONFIG')))


    my_model = MyModel(app)


    app.start(my_model)

    if my_model.app.appconfig.get_input_source().scheme == "https" and my_model.app.appconfig.get_autostart() == "True":
        logging.info("starting compute_ai, HTTPS and autostart")

        result = my_model.compute_ai(my_model.app.appconfig.get_input_source().geturl())
        my_model.app.fire_notification(result)


    # start Flask message handler here
    msg_handler = init_handler(app)

    flaskapp = msg_handler.get_flask_app()

    @flaskapp.route('/getCurrentFPS', methods=['GET'])
    def getFPS():
        return str(my_model.get_current_fps())

    @flaskapp.route('/getCurrentDelay', methods=['GET'])
    def getDelay():
        return str(my_model.get_current_video_delay())

    @flaskapp.route('/setMinimumFPS', methods=['GET'])
    def setMinimumFPS():
        minimum_fps = request.args.get('minimum_fps', default=30, type=int)
        if my_model.set_minimum_fps(minimum_fps):
            return "setting minimum fps to " + str(request.args.get('minimum_fps', default=30, type=int))
        else:
            return "did not change fps. They have to between 60 and 1"

    @flaskapp.route('/startAI', methods=['GET'])
    def startAI():
        minimum_fps = request.args.get('minimum_fps', default=30, type=int)
        return my_model.start_thread(minimum_fps)

    @flaskapp.route('/stopAI', methods=['GET'])
    def stopAI():
        return my_model.stop_thread()

    @flaskapp.route('/resetAI', methods=['GET'])
    def resetAI():
        my_model.load_ai_model("")
        return "AI model was reloaded"

    @flaskapp.route('/trainAI', methods=['POST'])
    def trainAI():
        global TRAINING
        logging.info("start training")
        if TRAINING:
            return "CURRENTLY TRAINING"

        TRAINING = True
        try:
            os.mkdir("train_data")
        except:
            pass
        logging.info(type(request.data))
        logging.info(len(request.data))
        with open("train_data/data.zip", "wb") as file:
            file.write(request.data)
        rez = train()
        TRAINING = False
        if rez:
            my_model.load_ai_model("a")
            return "SUCCESFULLY TRAINED MODEL"
        else:
            return "ERROR OCCURED WHILE TRAINING THE MODEL"

    flaskapp.run(host="0.0.0.0", threaded=True)


main()

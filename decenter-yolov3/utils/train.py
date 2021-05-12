#================================================================
#
#   File name   : train.py
#   Author      : PyLessons
#   Created date: 2020-08-06
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to train custom object detector
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import shutil
import numpy as np
import tensorflow as tf
from utils.dataset import Dataset
from utils.utils import load_yolo_weights, Create_Yolo, compute_loss, get_mAP

import yaml
import logging
import shutil


def process_data():
    shutil.unpack_archive("train_data/data.zip", "train_data", "zip")
    with open("train_data/train.yaml") as file:
        data = yaml.full_load(file)["annotations"]
    return data


def train():
    logging.info("started training")
    annotations = process_data()

    TRAIN_LOGDIR = "log"
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    if os.path.exists(TRAIN_LOGDIR):
        shutil.rmtree(TRAIN_LOGDIR)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    trainset = Dataset('train', "train_data/" + annotations["train"])
    testset = Dataset('test', "train_data/" + annotations["test"])

    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = 2 * steps_per_epoch
    total_steps = 22 * steps_per_epoch

    yolo = Create_Yolo(input_size=416, training=True, CLASSES="model_data/names.txt")
    try:
        yolo.load_weights(f"./model_data/yolov3_custom")
    except ValueError:
        logging.info("Shapes are incompatible, transfering Darknet weights")
        return False
    
    optimizer = tf.keras.optimizers.Adam()

    def train_step(image_data, target):
        TRAIN_LR_INIT = 2e-5
        TRAIN_LR_END = 1e-6
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0

            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES="model_data/names.txt")
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            global_steps.assign_add(1)
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=False)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES="model_data/names.txt")
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    mAP_model = Create_Yolo(input_size=416, CLASSES="model_data/names.txt") # create second model to measure mAP

    best_val_loss = 1000 # should be large at start
    for epoch in range(25):
        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            logging.info("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}".format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))

        if len(testset) == 0:
            logging.info("configure TEST options to validate model")
            yolo.save_weights(os.path.join("model_data", "yolo3_custom"))
            continue
        
        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()
            
        logging.info("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

    shutil.rmtree('/train_data')
    shutil.rmtree('/log')
    return True

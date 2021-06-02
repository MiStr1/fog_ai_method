# AI method for license plate detection

## Requires

docker

If you want to run the container with GPU you also need to setup the [container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Build

go to the decenter_base folder and run `docker build . -t dec-base-demo` 
(if you want to use the GPU version change the dockerfile there to use nvidia/cuda base).

Then go to the decenter-yolov3 folder and set up the MY_APP_CONFIG variable in the dockerfile. The input contains 
the URL of the input webm stream. The output contains the MQTT server URL and the topic where the data will be sent 
to. (You can also add  MY_APP_CONFIG later on when you run the container with `docker run -e MY_APP_CONFIG={json} ...`). 
Then run `docker build . -t yolo_license_plate`.
You can find prebuild images on dockerhub: [mihastravs/yolo_license_plate_gpu](https://hub.docker.com/repository/docker/mihastravs/yolo_license_plate_gpu), [mihastravs/yolo_license_plate](https://hub.docker.com/repository/docker/mihastravs/yolo_license_plate)

## Run

Run the image with `docker run -p=5000:5000 yolo_license_plate`.

Container can be accessed at http://localhost:5000


### Endpoints

http://localhost:5000/getCurrentFPS  returns current FPS of the object detection

http://localhost:5000/getCurrentDelay returns current delay of the stream

http://localhost:5000/setMinimumFPS sets minimum required FPS for the detection

http://localhost:5000/startAI starts the detection

http://localhost:5000/stopAI stops the detection

http://localhost:5000/resetAI resets the detection

http://localhost:5000/trainAI Retrains the method with new data.

### Retraining

Inside the demo_scripts folder, there is a script `test_train.py` to demonstrate how to retrain the method.
It requires requests python library which can be installed with `pip install requests`. It creates an archive
from the test_train folder and sends it to the container for training. Test_train contains two files `license.txt` 
and `license_test.txt`. The first contains the images for training and the second one for the validation.
Each line in those two files contains the image location and positions of license plates in the image in 
the standard yolo train file format.

## Retrieving detections through mqtt

Inside demo_scripts folder there is also a script `test_mqtt.py` which contains a demo script for data retrieval. At the 
bottom part of the script change the address and the topic of the MQTT server you are using. To run the script install
the paho library with `pip install paho-MQTT and` and opencv with `pip install OpenCV-python`. When this client receives 
the data from the docker container an image with bounding boxes is displayed and the rest of the data is outputed in a 
JSON form to the standard output.


## Serving file

If you need a stream to test the method we have also provided a servlet demo. It is contained inside the servlet folder.
Inside the folder folder add the webm videos on which you would like to test the method. Then build the servlet with
`docker build . -t servlet` command while you are inside the servlet folder. Make a bridge network net to connect 
discord containers then run the servlet with `docker run -p=8081:8081 --name=servlet --network=net -d servlet`. Also add 
`--network=net` when you are runing the yolo container. Yolo container should now find the stream at 
http://servlet:8081/filename.webm.
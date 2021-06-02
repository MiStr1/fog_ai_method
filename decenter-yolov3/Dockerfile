FROM dec-base-demo

# install tensorflow
#ARG TF_PACKAGE=tensorflow
#ARG TF_PACKAGE_VERSION=1.8
#RUN pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

#RUN pip --no-cache-dir install \
#    keras==2.1.5

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        python3-opencv \
    && apt-get install -y \
        tesseract-ocr \
    && apt-get install -y \
        libtesseract-dev

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /

# decenter package test code
ADD ./decenter_yolov3_0.1.zip /
ADD ./model_data /model_data
ADD ./main.py /
ADD utils /utils/
ADD ./MyModel.py /

WORKDIR /

ENV MY_APP_CONFIG="{\
  \"input\":{\
    \"url\":\"http://194.249.2.111:30090/cars_demo3.webm\"\
  },\
  \"output\":{\
    \"url\":{\
      \"mqtt\":\"mqtt://194.249.2.112:30533/jobs/67965ae8-d5e3-4e82-a123-151aae8bbc5e\"\
    }\
  },\
  \"ai_model\":{\
    \"url\":\"\",\
    \"model_name\":\"decenter_yolov3\",\
    \"model_version\":\"0.1\"\
  },\
  \"autostart\":{\
    \"value\":\"True\"\
  }\
}"


# Expose ports
# for flask
EXPOSE 5000

CMD ["python", "main.py"]

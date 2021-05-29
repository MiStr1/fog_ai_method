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

# Expose ports
# for flask
EXPOSE 5000

CMD ["python", "main.py"]

FROM tensorflow/tensorflow:latest-gpu

WORKDIR /tf-main

COPY requirements.txt requirements.txt

RUN pip install sacremoses --no-deps


RUN pip install -r requirements.txt

RUN pip install --no-deps keras-tuner
RUN pip install --force-reinstall protobuf==3.20

RUN pip freeze > requirements.txt

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
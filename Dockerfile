FROM python:3.7

WORKDIR /opt/app

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python setup.py install

CMD allennlp train -s $CHECKPOINT_PATH --include-package sftp config/fn.jsonnet

sudo docker build . -t "lome-training"

sudo docker run -v ~/public_html/hiaoxui-span-finder-e85cb04/model:/model -p 7749:7749 "lome-server"

sudo docker run --env-file .env -v /home/usuario/hiaoxui-span-finder-e85cb04/training:/training --gpus all "lome-training"

sudo docker run -p 7749:7749 "lome-fnbr-server"

sudo docker run -v ~/hiaoxui-span-finder-e85cb04/model:/model \
-v ~/hiaoxui-span-finder-e85cb04/inference:/inference \
--gpus all \
-it "lome-training" /bin/sh \
-c "pip install /inference/en_core_web_sm-3.2.0.tar.gz && python scripts/predict_mcgovern.py -m /model -i /inference/data/base_linkada_tokenizada.jsonl -o /inference/data/semantics.jsonl"

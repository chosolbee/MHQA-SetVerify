#!/bin/sh

mkdir -p .temp
mkdir -p data/raw

echo "\n\nDownloading raw HotpotQA data\n"
mkdir -p data/raw/hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O data/raw/hotpotqa/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json -O data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json
uv run python -m download.subsample_data --dataset hotpotqa --num-eval 1000 --num-test 1000

echo "\n\nDownloading raw 2WikiMultihopQA data\n"
mkdir -p data/raw/2wikimultihopqa
wget https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=0 -O .temp/2wikimultihopqa.zip
unzip -jo .temp/2wikimultihopqa.zip -d data/raw/2wikimultihopqa -x "*.DS_Store"
uv run python -m download.subsample_data --dataset 2wikimultihopqa --num-eval 1000 --num-test 1000

echo "\n\nDownloading raw MuSiQue data\n"
mkdir -p data/raw/musique
# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
uv run gdown "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&confirm=t" -O .temp/musique_data_v1.0.zip
unzip -jo .temp/musique_data_v1.0.zip -d data/raw/musique -x "*.DS_Store"
uv run python -m download.subsample_data --dataset musique --num-eval 1000 --num-test 1000

rm -rf .temp/

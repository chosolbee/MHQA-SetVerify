#!/bin/sh

mkdir -p .temp
mkdir -p data/corpus

echo "\n\nDownloading HotpotQA Wikipedia corpus (this will take ~5 mins)\n"
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -O .temp/hotpot_wikipedia.tar.bz2

echo "\n\nBuilding HotpotQA corpus\n"
mkdir -p data/corpus/hotpotqa
uv run python3 -m download.build_corpus --dataset hotpotqa --input-path .temp/hotpot_wikipedia.tar.bz2

echo "\n\nBuilding 2WikiMultiHopQA corpus\n"
mkdir -p data/corpus/2wikimultihopqa
uv run python3 -m download.build_corpus --dataset 2wikimultihopqa

echo "\n\nBuilding MuSiQue corpus\n"
mkdir -p data/corpus/musique
uv run python3 -m download.build_corpus --dataset musique

rm -rf .temp/

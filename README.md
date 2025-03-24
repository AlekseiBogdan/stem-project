# Building neural network for stemming music files

**Based on MoisesDB dataset** (https://github.com/moises-ai/moises-db)

Main goal of the project — create a neural network to automatically stem music files into distinctive music tracks.

## Prerequisites

- Python 3.11
- install requirements.txt ``pip install -r requirements.txt``

## Dataset

I chose MoisesDB dataset as an open-source easy-to-use dataset created for non-commercial use (see [this](https://arxiv.org/abs/2307.15913)).

MoisesDB is a comprehensive multitrack dataset for source separation beyond 4-stems, comprising 240 previously unreleased songs by 47 artists spanning twelve high-level genres. The total duration of the dataset is 14 hours, 24 minutes and 46 seconds, with an average recording length of 3:36 seconds ([Link](https://music.ai/research/)). 

## Benchmarking

To infer model use `infer.py` script via CLI

Just pass `python infer.py --input PATH_TO_YOUR_AUDIO_FILE` to your terminal, results will be saved in the `output_stems` folder (it will be created if was not before)

In case you want to specify output folder pass `--output` argument with path to the folder to the bash script

Also, you may want to use another model (after training one more or finetuning), in that case you can pass `--model` argument with path to your model


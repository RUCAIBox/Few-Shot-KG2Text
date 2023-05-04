# Few-Shot-KG2Text

This repository contains the source code and datasets for the ACL 2021 Findings paper "[Few-shot Knowledge Graph-to-Text Generation with Pretrained Language Models](https://arxiv.org/pdf/2106.01623.pdf)".

# Directory

- [Requirements](#Requirements)
- [Datasets and Models](#Datasets-and-Models)
- [Training Instructions](#Training-Instructions)
- [Testing Instructions](#Testing-Instructions)
- [License](#License)
- [Reference](#References)

# Requirements

- Python 3.7
- Pytorch 1.8
- torch-geometric 
- transformers
- Anaconda3

# Datasets and Models
Our processed KG-to-Text datasets, including WebNLG, Agenda and GenWiki-Fine, can be downloaded from [Google drive](https://drive.google.com/drive/folders/10W1-Sc8snZJU6_-0uYXOWWGhO6OaU4AO?usp=sharing). Specifically, we choose the three largest domains, i.e., Food, Airport, and Building, from WebNLG, and two largest domains, i.e., Sports and Games, from GenWiki-Fine. Their original datasets can be downloaded from [WebNLG](https://webnlg-challenge.loria.fr/challenge_2017/), [Agenda](https://github.com/rikdz/GraphWriter), and [Genwiki](https://github.com/zhijing-jin/genwiki). The WebNLG dataset is made of xml files, you can refer to [this repository](https://github.com/zhijing-jin/WebNLG_Reader) to transform them into json files. We utilize the large version of BART from Hugging Face.

# Training Instructions

Before implement our code, you need to pre-process each dataset (train.json, valid.json, and test.json), and build vocabulary and initilized node embeddings:

```
python process.py
python build_vocab.py
python build_embedding.py
```

Besides, we utilize [TransE](https://github.com/thunlp/OpenKE) for obtaining the KG embeddings for RBFS. After preparing all the files, you can run the run.sh file to start training by setting ```mode: train``` in config.yaml.

```
sh run.sh 
```

# Testing Instructions

At the test stage, you can run the run.sh file to start testing by setting ```mode: test``` in config.yaml.

# License

```
License agreement
This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions. 
2. That you include a reference to the dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our References; for other media cite our preferred publication as listed on our website or link to the dataset website.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (Wayne Xin Zhao, School of Information, Renmin University of China).
```

# References

If this work is useful in your research, please cite our paper.

```
@inproceedings{junyi2021KG2Text,
  title={{F}ew-shot {K}nowledge {G}raph-to-{T}ext {G}eneration with {P}retrained {L}anguage {M}odels},
  author={Junyi Li, Tianyi Tang, Wayne Xin Zhao, Zhicheng Wei, Nicholas Jing Yuan, Ji-Rong Wen},
  booktitle={ACL Findings},
  year={2021}
}
```


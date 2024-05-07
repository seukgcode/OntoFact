# OntoFact: Unveiling Fantastic Fact-Skeleton of LLMs via Ontology-Driven Reinforcement Learning
This is the source code of our AAAI 2024 paper "OntoFact: Unveiling Fantastic Fact-Skeleton of LLMs via Ontology-Driven Reinforcement Learning"
[Paper](https://doi.org/10.1609/aaai.v38i17.29859)

## Quick Links
- [OntoFact: Unveiling Fantastic Fact-Skeleton of LLMs via Ontology-Driven Reinforcement Learning](#OntoFact-Unveiling-Fantastic-Fact-Skeleton-of-LLMs-via-Ontology-Driven-Reinforcement-Learning)
  - [Quick Links](#quick-links)
  - [Overview](#overview)
- [Usage](#usage)
  - [Getting Started](#getting-started)
    - [Environment Installation](#environment-installation)
    - [Data Preprocessing](#data-preprocessing)
  - [Training \& Evaluation](#training--evaluation)
  - [Q\&A](#qa)
- [Citation](#citation)

## Overview
We propose OntoFact, a novel adaptive framework for detecting unknown facts in LLMs, dedicated to mining the ontology-level skeleton of the missing knowledge. 
The following figure is an illustration of our methods.

![](_doc/framework.png)

# Usage

## Getting Started

The structure of the folder is shown below:

```csharp
 OntoFact
 ├─KG_embedding
 ├─LLMs_Factuality_Evaluation
 ├─Ontology-Driven_Reinforcement_Learning
 ├─Dataset_Process_Code
 ├─requirements.txt
 └README.md
```

Introduction to the structure of the folder:

- /KG_embedding: Source code of knowledge graph embedding (KGE) for training in 5 datasets.
- /LLMs_Factuality_Evaluation: Source code of factual evaluation of 32 LLMs on 5 datasets using predefined prompt templates.
- /Ontology-Driven_Reinforcement_Learning: Source code of the ontology-driven reinforcement (ORL) learning.
- /Dataset_Process_Code: Source code of the 5 benchmarks built.

### Environment Installation

See `requirements.txt`

<details>
<summary>For training and limited evaluation</summary>

```bash
# python >= 3.9
# Basic pytorch environment, if different LLMs require different versions, please substitute as appropriate. 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# When python >= 3.10, please refer to [Link](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss#compiling-the-python-interface-within-an-anaconda-install) install faiss-gpu. 
conda install -c pytorch faiss-gpu==1.7.3
pip install transformers tqdm
```

</details>

<details>
<summary>For evaluation</summary>

```bash
# -- Prepare/Train KG Embeddings --
# 1. Download all benchmarks (DBpedia, YAGO, CN-DBpedia, BIOS 2.2 (ENG), BIOS 2.2 (CHS)) from [Google Drive](https://drive.google.com/drive/folders/1vqPhgdISICLs-yPi6OTBg3Ik9D0YyGuk?usp=drive_link) to ./KG_embedding/data
# 2. Run the code with this in the shell:
cd ./KG_embedding
sh ./train.sh
# 3. Wait for the training to finish or simply download the trained embedded file from [Google Drive](https://drive.google.com/drive/folders/1vqPhgdISICLs-yPi6OTBg3Ik9D0YyGuk?usp=drive_link) to ./KG_embedding/model. 
# 4. Run the code with this in the shell: (Then you will obtain the embeddings of isntance and ontology graph in the current directory)
cd ./KG_embedding
python ./KG_embedding/generate_embedding_npy.py

pip install pandas

# -- STS --
# 1. Get code from https://github.com/princeton-nlp/SimCSE
# 2. Install SentEval
git clone https://github.com/princeton-nlp/SimCSE.git
# find file "SimCSE/SentEval/senteval/sts.py"
# Modify lines 42 and 43 of the code to read as follows:
# <42> sent1 = np.array([s.split() for s in sent1], dtype=object)[not_empty_idx]
# <43> sent2 = np.array([s.split() for s in sent2], dtype=object)[not_empty_idx]
cd SimCSE/SentEval
pip install .
pip install prettytable
```

</details>

## Training & Evaluation

Training Scripts:
```bash
cd ./LLMs_Factuality_Evaluation
sh train.sh
```

Evaluation ORL Scripts:
```bash
# 1. Download all processed data from [Google Drive](https://drive.google.com/drive/folders/1vqPhgdISICLs-yPi6OTBg3Ik9D0YyGuk?usp=drive_link) to ./Ontology-Driven_Reinforcement_Learning/data
# 2. Run the code with this in the shell
cd ./Ontology-Driven_Reinforcement_Learning
sh train.sh
```
## Q&A
NOTE: Due to time constraints, the submitted code is code that has not been refactored, so in some cases it may contain some bugs that we didn't catch, but that doesn't affect the results in our paper.

If you have any questions, please submit an [issue](https://github.com/seukgcode/OntoFact/issues/new) or contact ziyus1999\<AT\>gmail.com or ziyus1999\<AT\>seu.edu.cn.

> 1. Datasets and Evaluation Detailed Results can be found at this link: [google drive](https://drive.google.com/drive/folders/1vqPhgdISICLs-yPi6OTBg3Ik9D0YyGuk?usp=drive_link)


# Citation

If you find this method or code useful, please cite

```bibtex
@inproceedings{shang2024ontofact,
  title={Ontofact: Unveiling fantastic fact-skeleton of llms via ontology-driven reinforcement learning},
  author={Shang, Ziyu and Ke, Wenjun and Xiu, Nana and Wang, Peng and Liu, Jiajun and Li, Yanhui and Luo, Zhizhao and Ji, Ke},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={17},
  pages={18934--18943},
  year={2024}
}
```
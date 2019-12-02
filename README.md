# mse.tm.chatbot.base
Chatbot - QA Component - Master's Thesis at Master of Engineering (MSE), Switzerland

## About
This repository is used as a component of the Master's Thesis of Romain Claret.

## What
This component is a question answering chatbot.

## Requirements
- Python 2 or 3
- Install the python requirements
 ```shell
 conda/pip install spacy requests pybind11 hdt networkx
 ```
- Install the spacy model: 
  ```shell
  python -m spacy download en_vectors_web_lg
  ```
- Get the submodules
 ```shell
 git submodule init
 git submodule update
 ```
- Download the data
```shell
  bash initialize.sh
```

### Errors
- If something goes wrong with the hdt library, check out this link https://pypi.org/project/hdt/, usually the fix is to install the Python Development headers.


## Modules used
- [CONVEX fork](https://github.com/RomainClaret/CONVEX) is used for the subgraph qa system.
  - Paper: Look before you Hop: Conversational Question Answering over Knowledge Graphs Using Judicious Context Expansion
  - Authors: Philipp Christmann, Rishiraj Saha Roy, Abdalghani Abujabal, Jyotsna Singh, and Gerhard Weikum, CIKM 2019.
  - Original github repo: https://github.com/PhilippChr/CONVEX
  - License: MIT license by Philipp Christmann
  - Website: http://qa.mpi-inf.mpg.de/convex/
  - PDF preprint of the CIKM'19 paper: https://arxiv.org/abs/1910.03262
 
## License
This project by Romain Claret is licensed under the MIT license. Note that used modules (see above) may be licensed differently; however, the license capabilities are maintained.

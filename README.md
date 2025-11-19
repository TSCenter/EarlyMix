# EarlyMix

Source code (PyTorch) and dataset of the paper "[EarlyMix: Hierarchical Mixing for Early Time Series Classification](https://ieeexplore.ieee.org/document/11209452)", which is accepted by IEEE International Conference on Multimedia & Expo 2025 (ICME 2025).


## Dataset
You can download the dataset from [here](https://www.timeseriesclassification.com/dataset.php).

Unzip the dataset and put it in the `script/dataset` folder.

The file structure is as follows:
```
script/dataset/
    ChlorineConcentration/
    .../
    ECG200/
```


## Requirements
You can install the requirements by running the following command:
```bash
pip install -r requirements.txt
```

## Running

You can use the following command:

```bash
bash run_earlyts_all.sh
```


## License
If you use EarlyMix in a scientific publication, we would appreciate citations to the following paper:
```
@inproceedings{DBLP:conf/acl/HuHZ25,
  author       = {Shuguo Hu and
                  Jun Hu and
                  Huaiwen Zhang},
  editor       = {Wanxiang Che and
                  Joyce Nabende and
                  Ekaterina Shutova and
                  Mohammad Taher Pilehvar},
  title        = {Synergizing LLMs with Global Label Propagation for Multimodal Fake
                  News Detection},
  booktitle    = {Proceedings of the 63rd Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2025, Vienna, Austria,
                  July 27 - August 1, 2025},
  pages        = {1426--1440},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.acl-long.72/},
  timestamp    = {Sun, 02 Nov 2025 21:27:24 +0100},
  biburl       = {https://dblp.org/rec/conf/acl/HuHZ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

License: GPLv3

Copyright (c) 2024-2025 IMU, China & HFUT, China & NUS, Singapore.

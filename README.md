# EarlyMix

Source code (PyTorch) and dataset of the paper "[EarlyMix: Hierarchical Mixing for Early Time Series Classification](https://ieeexplore.ieee.org/document/11209452)", which is accepted by IEEE International Conference on Multimedia & Expo 2025 (ICME 2025).

## Homepage and Paper
Homepage: [https://github.com/TSCenter/EarlyMix](https://github.com/TSCenter/EarlyMix)

Paper Access: IEEE [https://ieeexplore.ieee.org/document/11209452](https://ieeexplore.ieee.org/document/11209452)

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
@inproceedings{DBLP:conf/icmcs/HuHLZ25,
  author       = {Shuguo Hu and
                  Jun Hu and
                  Junwei Lv and
                  Huaiwen Zhang},
  title        = {EarlyMix: Hierarchical Mixing for Early Time Series Classification},
  booktitle    = {{IEEE} International Conference on Multimedia and Expo, {ICME} 2025,
                  Nantes, France, June 30 - July 4, 2025},
  pages        = {1--6},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/ICME59968.2025.11209452},
  doi          = {10.1109/ICME59968.2025.11209452},
  timestamp    = {Fri, 07 Nov 2025 11:48:59 +0100},
  biburl       = {https://dblp.org/rec/conf/icmcs/HuHLZ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

License: GPLv3

Copyright (c) 2024-2025 IMU, China & HFUT, China & NUS, Singapore.

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
@INPROCEEDINGS{11209452,
  author={Hu, Shuguo and Hu, Jun and Lv, Junwei and Zhang, Huaiwen},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={EarlyMix: Hierarchical Mixing for Early Time Series Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Measurement;Time series analysis;Refining;Predictive models;Feature extraction;Harmonic analysis;Data models;Reliability;Synthetic data;time series;mixing;early classification},
  doi={10.1109/ICME59968.2025.11209452}}
```

License: GPLv3

Copyright (c) 2024-2025 IMU, China & HFUT, China & NUS, Singapore.

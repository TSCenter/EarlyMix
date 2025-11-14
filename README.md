# EarlyMix

Source code (PyTorch) and dataset of the paper "[EarlyMix: Hierarchical Mixing for Early Time Series Classification](https://www.computer.org/csdl/proceedings-article/icme/2025/11209452/2beCG8sJoEo)", which is accepted by IEEE International Conference on Multimedia & Expo 2025 (ICME 2025).


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

License: GPLv3

Copyright (c) 2024-2025 IMU, China & HFUT, China & NUS, Singapore.

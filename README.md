## Car plates project. Segmentation model (part 1/3)

Neural network segmentation model to select car plate area with number. 


### Dataset

Dataset include 1754 images of car with plates in `jpg` from 4 countries (include Russia) in COCO format.

I use several open data from kaggle to compile single dataset:

* [Automatic-Number-Plate-Recognition](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection)
* [Car-and-License-Plate-Detection](https://www.kaggle.com/datasets/riotulab/car-and-license-plate-detection)
* [Car-License-Plate-Detection](https://www.kaggle.com/datasets/amirhoseinahmadnejad/car-license-plate-detection-iran)
* [Car-plate-object-detection](https://www.kaggle.com/datasets/andrewteplov/car-plate-object-detetcion)

On each image was selected box with car plate and for some data was selected box with car (`plate` and `car` classes).
 
Notes: 

* For some data only clearly visible car plates were selected. 
* Example of data you can see in [notebook](notebooks/EDA.ipynb).
* Model inference and export you can see in [notebook](notebooks/inference_convert.ipynb)

To download data:

```shell
make download_dataset
```


### Environment setup

1. Create and activate python venv
    ```shell
    python3 -m venv venv
    . venv/bin/activate
    ```

2. Install libraries
   ```shell
    make install
   ```
   
3. Run linters
   ```shell
   make lint
   ``` 

4. Tune [config.yaml](configs/config.yaml)

5. Train
   ```shell
   make train
   ```


### Additional information

* Inference example in [notebook](notebooks/inference_convert.ipynb)
* [Best experiment in ClearML](https://app.clear.ml/projects/8a4a72ee644148f781e5ba6beaaf8c65/experiments/36cd7b5e58ad490ca74676ffd11577ec/output/execution)
* [History of experiments](HISTORY.md)

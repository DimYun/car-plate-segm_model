## Домашняя работа №1

Решение задачи по определению положения номерного знака на автомобиле. 

### Датасет

Включает 40479 изображений автомобилей с номерными знаками в формате `jpg` 
из более 5 стран (включая Россию).

Для обучения модели использовались открытые данные с kaggle:

* [Automatic-Number-Plate-Recognition](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection)
* [Car-and-License-Plate-Detection](https://www.kaggle.com/datasets/riotulab/car-and-license-plate-detection)
* [Car-License-Plate-Detection](https://www.kaggle.com/datasets/amirhoseinahmadnejad/car-license-plate-detection-iran)
* [Car-plate-object-detection](https://www.kaggle.com/datasets/andrewteplov/car-plate-object-detetcion)

На каждом изображении выделен бокс с номерным знаком и на некоторых данных 
выделен автомобиль (`plate` и `car` соответственно).
 
На части данных не выделены все номерные знаки, а только самые близкие и  
читаемые. Сами данные объединили в единый датасет в СОСО  формате и 
загрузили в CVAT на проверку, валидацию и доразметку.  Процедуру обработки 
данных и создания единого датасета можно посмотреть [в тетрадке]().

Скачать датасет:

```bash
make download_dataset
```

### Подготовка окружения

1. Создание и активация окружения
    ```bash
    python3 -m venv venv
    . venv/bin/activate
    ```

2. Установка библиотек
   ```
    make install
   ```
   
3. Запуск линтеров
   ```
   make lint
   ``` 

4. Логи лучшего эксперимента в ClearML

- 

5. Настраиваем [config.yaml](configs/config.yaml) под себя.


### Обучение

Запуск тренировки:

```bash
make train
```

### Инеренс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/inference.ipynb).

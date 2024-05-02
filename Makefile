install:
	pip install -r requirements.txt

download_dataset:
	wget -O dataset.zip "https://www.kaggle.com/datasets/nikitarom/planets-dataset/download?datasetVersionNumber=3"
	unzip dataset.zip
	rm dataset.zip

upload_weit:
	rsync -aP experiments user@test_server:/home/user/experiments

train:
	PYTHONPATH=. python src/train.py configs/config.yaml

lint:
	PYTHONPATH=. black train.py src
	PYTHONPATH=. nbstripout notebooks/*.ipynb
	PYTHONPATH=. tox
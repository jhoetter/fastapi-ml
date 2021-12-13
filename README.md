# fastapi-ml
This is an example project to set up a ML service using [FastAPI](https://fastapi.tiangolo.com/).

## Installation
To get this app up and running, simply create a virtual environment (e.g. using Anaconda) and run the following int your activated environment: `$ pip install -r requirements.txt` .

## Retrieve the data
The data `sap_press.json` is available at this [Kaggle account](https://www.kaggle.com/johoetter). Store it in this repository.

## Run the training file
Execute `$ python training.py`, which will run the training procedure to create `my_model.pkl`. This file is NOT designed to create a great model, but to showcase how an example scikit-learn model can be build and stored easily to a pickle file. You will build much greater models in your use case 💯.

*Changing the configs*: You can adapt `train_configs.yaml`, but this should not be needed in general.

## Running the service
Once you have created the `my_model.pkl` file, you can get the service up and running via `$ uvicorn api:app`. In dev mode, you can also run `$ uvicorn api:app --reload`, which will reload the service on save.
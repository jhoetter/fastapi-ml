import yaml
import pickle
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
with open("train_configs.yaml", "r") as yaml_file:
    configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

embedder = SentenceTransformer(configs["embedding_config"])
with open(configs["save_to"], "rb") as file:
    clf = pickle.load(file)


@app.post("/topic_classification/{press_release}")
def predict_topic(press_release: str):
    embedded_press_release = embedder.encode(press_release).reshape(1, -1)
    prediction = clf.predict(embedded_press_release)[0]
    return prediction

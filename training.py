import pandas as pd
import pickle
import yaml
from wasabi import msg

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    msg.info("Starting training program")

    with open("train_configs.yaml", "r") as yaml_file:
        configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

    msg.info(f'Using {configs["embedding_config"]} for embeddings')
    embedder = SentenceTransformer(configs["embedding_config"])

    msg.info(f'Reading data from {configs["embedding_config"]}')
    df = pd.read_json(configs["json_input"])
    X = df[configs["input_attribute"]]
    y = df[configs["label_attribute"]]

    msg.info(f"Starting embedding procedure on {len(df)} input records")
    X_emb = embedder.encode(X)

    msg.info("Training logistic regression model")
    clf = LogisticRegression()
    clf.fit(X_emb, y)

    msg.info(f'Storing trained model to {configs["save_to"]}')
    with open(configs["save_to"], "wb") as file:
        pickle.dump(clf, file)

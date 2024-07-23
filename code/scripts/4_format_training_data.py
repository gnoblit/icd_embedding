
import polars as pl

def main(pos_path: str='/home/gnoblit/takehome/codametrix/data/clean/positive_train_data.ndjson', neg_path: str='/home/gnoblit/takehome/codametrix/data/clean/negative_train_data.ndjson'):
    """Function exists to format training data for triplet loss function. From SentenceTransformers: "Given a triplet of (anchor, positive, negative), the loss minimizes the distance between anchor and positive while it maximizes the distance between anchor and negative." 

    Takes in path (str) to positive and negative training examples
    Concats these datasets 
    """

    pos_df = pl.read_ndjson(pos_path).sort('code')
    neg_df = pl.read_ndjson(neg_path)


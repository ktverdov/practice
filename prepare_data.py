import os
import numpy as np
import pandas as pd
import json

import argparse


def prepare_ru_tweet(data_dir):
    negative = pd.read_csv(os.path.join(data_dir, "negative.csv"), sep=";", header=None, usecols=[3, 4])
    negative.columns = ["text", "sentiment"]

    positive = pd.read_csv(os.path.join(data_dir, "positive.csv"), sep=";", header=None, usecols=[3, 4])
    positive.columns = ["text", "sentiment"]

    df = negative.append(positive).reset_index(drop=True)

    split = np.random.rand(len(df)) < 0.8
    split = ["train" if x else "val" for x in split]

    df["split"] = split
    df["source"] = "tweet"

    return df


def prepare_ru_sent(data_dir):
    def preprocess(df, split):
        df = df[df["label"].isin(["positive", "negative"])]
        df = df.rename(columns={"label": "sentiment"})
        df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": -1})
        df["split"] = split

        return df

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train = preprocess(train, "train")
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test = preprocess(test, "val")
    
    df = train.append(test).reset_index(drop=True)
    df["source"] = "sent"
    
    return df


def prepare_ru_news(data_dir):

    with open(os.path.join(data_dir, "train.json")) as f:
        df = json.load(f)
    
    df = pd.json_normalize(df)
    df = df[["text", "sentiment"]]
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": -1})
    
    split = np.random.rand(len(df)) < 0.8
    split = ["train" if x else "val" for x in split]
    df["split"] = split
    df["source"] = "news"
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweet_dir', default="", type=str)
    parser.add_argument('--sent_dir', default="", type=str)
    parser.add_argument('--news_dir', default="", type=str)
    parser.add_argument('--out_file', type=str, required=True)
    
    return parser.parse_args()


def prepare_data(args):
    ru_tweet = prepare_ru_tweet(args.tweet_dir)
    ru_sent = prepare_ru_sent(args.sent_dir)
    ru_news = prepare_ru_news(args.news_dir)
    
    data = pd.concat([ru_tweet, ru_sent, ru_news], ignore_index=True)
    data.to_csv(args.out_file, index=False)
    

if __name__ == "__main__":
    np.random.seed(42)
    args = parse_args()
    prepare_data(args)

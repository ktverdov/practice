
#convert raw data to dataset
python prepare_data.py --tweet_dir "/home/ktverdov/education/gos/data/ru_tweet/" \
    --sent_dir "/home/ktverdov/education/gos/data/ru_sent/" \
    --news_dir "/home/ktverdov/education/gos/data/ru_news/" \
    --out_file "/home/ktverdov/education/gos/data/full_data.csv"

python train_baseline/train.py --data_file "/home/ktverdov/education/gos/data/full_data.csv" \
    --tf_checkpoint "./checkpoints/tfidf.dump" \
    --model_checkpoint "./checkpoints/logreg.dump"
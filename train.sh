
#convert raw data to dataset
# python prepare_data.py --tweet_dir "/home/ktverdov/education/gos/data/ru_tweet/" \
#     --sent_dir "/home/ktverdov/education/gos/data/ru_sent/" \
#     --news_dir "/home/ktverdov/education/gos/data/ru_news/" \
#     --out_file "/home/ktverdov/education/gos/data/full_data.csv"

# python train_baseline/train.py --data_file "/home/ktverdov/education/gos/data/full_data.csv" \
#     --tf_checkpoint "./checkpoints/tfidf.dump" \
#     --model_checkpoint "./checkpoints/logreg.dump"
    
python train_bert/train.py --data_path "/home/ktverdov/education/gos/data/full_data.csv" \
    --tokenizer_checkpoint '/home/ktverdov/education/gos/code/bert_pretrained/ru_conversational_cased_L-12_H-768_A-12_pt/vocab.txt' \
    --model_checkpoint '/home/ktverdov/education/gos/code/bert_pretrained/ru_conversational_cased_L-12_H-768_A-12_pt/'\
    --out_dir "./trained_bert"
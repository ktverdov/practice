import numpy as np
import pandas as pd

import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import argparse
from tqdm import tqdm

from utils import seed_everything


def get_accuracy(preds, labels):
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    return np.sum(preds == labels) / len(labels)


class Trainer():
    def __init__(self):
        pass
    
    def set_parameters(self, parameters):
        self.model = parameters["model"]
        self.optimizer = parameters["optimizer"]
        self.datasets = parameters["datasets"]
        self.epochs = parameters["epochs"]
        self.batch_size = parameters["batch_size"]
        self.out_dir = parameters["out_dir"]
        
    
    def train(self):
        self.train_dataloader = DataLoader(self.datasets["train"], 
                                           sampler=RandomSampler(self.datasets["train"]), 
                                           batch_size=self.batch_size)
        
        self.val_dataloader = DataLoader(self.datasets["val"], 
                                         sampler=SequentialSampler(self.datasets["val"]), 
                                         batch_size=self.batch_size)
        
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps = 0, 
                                                         num_training_steps = len(self.train_dataloader) * self.epochs)

        
        best_score = 0
        
        for epoch in range(0, self.epochs):
            self._train_epoch()
            
            curr_score = self._evaluation()
            
            if curr_score > best_score:
                best_score = curr_score
                
                self.model.save_pretrained(self.out_dir)

    
    def _train_epoch(self):
        total_loss = 0
    
        self.model.train()

        for step, batch in tqdm(enumerate(self.train_dataloader)):
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].cuda()
            b_labels = batch[2].cuda()

            self.model.zero_grad()        

            outputs = self.model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)
            loss = outputs[0]
            loss.backward()

            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            self.scheduler.step()

        avg_train_loss = total_loss / len(self.train_dataloader)            

        print("Average training loss: {0:.2f}".format(avg_train_loss))
        
        return avg_train_loss

    
    def _evaluation(self):
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in tqdm(self.val_dataloader):
            batch = tuple(t.cuda() for t in batch)

            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():        
                outputs = self.model(b_input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = get_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        
        return eval_accuracy/nb_eval_steps
    

def prepare_data(df, tokenizer, max_len):
    sentences = df.text.values

    input_ids = []

    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)

        input_ids.append(encoded_sent)

    input_ids = pad_sequences(input_ids, 
                              maxlen=max_len, 
                              dtype="long", 
                              value=0, 
                              truncating="post", 
                              padding="post")

    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(df.sentiment.values)
    attention_masks = torch.tensor(attention_masks)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    return dataset    

def load_data(data_path, tokenizer, max_len):
    data = pd.read_csv(data_path)
    data.loc[data.sentiment == -1, "sentiment"] = 0
    
    datasets = dict()
    datasets["train"] = prepare_data(data[data["split"] == "train"], tokenizer, max_len)
    datasets["val"] = prepare_data(data[data["split"] == "val"], tokenizer, max_len)
    
    return datasets
    
    
def train(data_path, tokenizer_checkpoint, model_checkpoint, out_dir):
    max_len = 128
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_checkpoint, 
                                              do_lower_case=True)
                        
    datasets = load_data(data_path, tokenizer, max_len)
    
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_checkpoint,
                                                          num_labels = 2, 
                                                          output_attentions = False, 
                                                          output_hidden_states = False,)
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    model.cuda()

    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
   
    epochs = 4
    batch_size = 32
    
    trainer = Trainer()
    trainer_parameters = {"datasets": datasets, 
                          "model": model, 
                          "optimizer": optimizer, 
                          "epochs": epochs, 
                          "batch_size": batch_size, 
                          "out_dir": out_dir
                         }
    
    trainer.set_parameters(trainer_parameters)
    trainer.train()
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--tokenizer_checkpoint', type=str, required=True)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    
    return parser.parse_args()    
    

if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)
    train(args.data_path, args.tokenizer_checkpoint, args.model_checkpoint, args.out_dir)
                       
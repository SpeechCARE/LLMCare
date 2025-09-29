from utils import *
import pandas as pd
import numpy as np 
import torch
import torch.nn
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from Model import *

class Alzheimer_text_classification():

    def __init__(self, configs, train_data=None, valid_data=None, test_data=None, phase='training'):
        self.max_seq_len = configs['max_seq_len']
        self.batch_size = configs['batch_size']
        self.epochs = configs['epochs']
        self.lr = configs['learning_rate']
        self.last_n_layers = configs['last_n_layers_list']
        self.path_max_f1 = configs['path_to_save_weights_based_f1']
        self.extended = configs['extended']
        self.fusion_type = configs['fusion_type']
        self.sub_inter_fusion = configs['sub_inter_fusion']
        self.trans_mid_size = configs['trans_mid_size']
        self.ling_input_size = configs['ling_input_size']
        self.ling_mid_size = configs['ling_mid_size']
        self.liwc_input_size = configs['liwc_input_size']
        self.liwc_mid_size = configs['liwc_mid_size']
        self.transformers_name = configs['transformers_name']
        self.fine_tuning = configs['fine-tuning']
        self.embedding_output_structure = configs['embedding_output_structure']
        self.jmim = configs['jmim']
        self.seed = configs['seed']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flag = 0

        # Load the transformers tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformers_name)

        if phase=='training':
            self.train_dataloader = self.tokenize_and_dataloader(train_data, data_type='train')
            self.valid_dataloader = self.tokenize_and_dataloader(valid_data, data_type='valid')
        elif phase=='testing':
            self.test_dataloader = self.tokenize_and_dataloader(test_data, data_type='test')

    def tokenize_and_dataloader(self, data, data_type):

        texts, labels = data['text'], pd.get_dummies(data['label'])

        ling_features = data[[col for col in data.columns if 'LING' in col]]
        liwc_keyword = 'JMIM' if self.jmim else 'LIWC'
        LIWC_features = data[[col for col in data.columns if liwc_keyword in col]]

        # Tokenize and encode sequences in the training set
        tokens = self.tokenizer.batch_encode_plus(texts.tolist(), max_length=self.max_seq_len,
                                                  pad_to_max_length=True,truncation=True, return_token_type_ids=False)

        # Wrap tensors
        data = TensorDataset( torch.tensor(tokens['input_ids']),
                              torch.tensor(tokens['attention_mask']),
                              torch.tensor(labels.to_numpy()),
                              torch.tensor(ling_features.to_numpy(), dtype=torch.float32),
                              torch.tensor(LIWC_features.to_numpy(), dtype=torch.float32))

        if data_type=='train':
            # sampler for sampling the data during training
            sampler = RandomSampler(data)
        else :
            # sampler for sampling the data during validating and testing
            sampler = SequentialSampler(data)

        # DataLoader for dataset
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)


    def creating_model(self, inference=False):

        model = Network(self.transformers_name, self.embedding_output_structure, self.seed,
                        extended=self.extended, fusion_type=self.fusion_type, sub_inter_fusion=self.sub_inter_fusion,
                        trans_mid_size=self.trans_mid_size, ling_input_size=self.ling_input_size,
                        ling_mid_size=self.ling_mid_size, liwc_input_size=self.liwc_input_size,
                        liwc_mid_size=self.liwc_mid_size).to(self.device)

        if not inference:
            if self.fine_tuning == 'No fine-tuning':
                model.freeze_all()
            elif self.fine_tuning == 'All layers':
                model.finetune_all()
            elif self.fine_tuning == 'Last_n_layers_of_transformers':
                all_layers = 6 if self.transformers_name in ['distilbert-base-uncased', 'distilroberta-base'] else 12
                self.flag = model.finetune_n_last_layers(self.last_n_layers, all_layers)
            else:
                print("\nNote: configs['fine-tuning'] should be one of two phrases('No fine-tuning' or 'All layers' or 'Last_n_layers_of_transformers').\n")
                sys.exit(0)
            if self.flag == 1:
                return

        return model


    def train(self, model, optimizer, loss_function, dataloader, epoch):
        model.train()

        # empty lists to save model predictions
        pred_labels, true_labels, losses = [], [], []
        # iterate over batches
        for step, batch in enumerate(dataloader):

            # push the batch to gpu
            (sent_id, mask, labels, ling_features, LIWC_features) = [t.to(self.device) for t in batch]

            # clear previously calculated gradients
            model.zero_grad()

            # get model predictions for the current batch
            outputs = model(sent_id, mask, ling_features, LIWC_features)

            #compute the loss between actual and predicted values
            loss = loss_function(outputs, labels.float())
            losses.append(loss.item())

            # backward pass to calculate the gradients
            loss.backward()

            # update parameters
            optimizer.step()

            pred_labels.extend(torch.argmax(outputs, dim = 1).cpu().numpy())
            true_labels.extend(torch.argmax(labels.double(), dim=1).cpu().numpy())


            if step % round((len(self.train_dataloader) / 5)) == 0:
                print(f'\r[Epoch][Batch] = [{epoch + 1}][{step}] -> Loss = {np.mean(losses):.4f} ')

        #returns the loss and predictions
        return round(np.mean(losses),4) , pred_labels, true_labels


    def evaluate(self, model, loss_function, dataloader):
        print("\nEvaluating...")

        # deactivate dropout layers
        model.eval()

        # empty list to save the model predictions
        pred_probs, pred_labels, true_labels, losses = [], [], [], []

        # iterate over batches
        for step, batch in enumerate(dataloader):

            # push the batch to gpu
            (sent_id, mask, labels , ling_features, LIWC_features) = [t.to(self.device) for t in batch]

            # deactivate autograd
            with torch.no_grad():
                # model predictions
                outputs = model(sent_id, mask , ling_features, LIWC_features)
                outputs_prob = F.softmax(outputs, dim=1)

                # compute the validation loss between actual and predicted values
                loss = loss_function(outputs, labels.float())
                losses.append(loss.item())

                pred_probs.extend(outputs_prob.detach().cpu().numpy()[:, 1])
                pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                true_labels.extend(torch.argmax(labels.double(), dim=1).cpu().numpy())

        return round(np.mean(losses), 4), pred_probs, pred_labels, true_labels


    def start_training_and_validating(self):

        model = self.creating_model()
        loss_function = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=2e-3)  # 3e-5 not good

        # set initial loss to infinite
        best_valid_f1 = -1
        # empty lists to store training and validation loss of each epoch
        loss_list, f1_list = [], []

        #for each epoch
        print('\n' ,'.' * 100,)
        print('Start Training ....', end = '')
        for epoch in range(self.epochs):

            # Train the model
            train_loss, train_pred_labels, train_true_labels  = self.train(model, optimizer, loss_function, self.train_dataloader , epoch)

            # Evaluate the model
            valid_loss, _, valid_pred_labels, valid_true_labels = self.evaluate(model, loss_function, self.valid_dataloader)

            train_f1 = f1_score(train_true_labels, train_pred_labels)
            train_f1 = round(train_f1*100, 3)

            valid_f1 = f1_score(valid_true_labels, valid_pred_labels)
            valid_f1 = round(valid_f1*100, 3)

            #save the best model based on maximum f1-score
            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                self.best_epoch_f1 = epoch+1
                torch.save(model.state_dict(), self.path_max_f1)

            loss_list.append([train_loss, valid_loss])
            f1_list.append([train_f1, valid_f1])

            print('------------->Train -> Loss =',train_loss,'f1-score =',train_f1 ,
                  '\n------------->VAl -> Loss =',valid_loss,'f1-score =',valid_f1,'\n')

        print(f'Max F1 on test: {np.max(np.array(f1_list)[:, 1])} - epoch: {self.best_epoch_f1}')
        plot_training(np.array(loss_list), np.array(f1_list), ' ')

    def testing(self, path_model, loader_type):
        model = self.creating_model(inference=True)
        #load weights of best model
        model.load_state_dict(torch.load(path_model))
        # get predictions for test data
        if loader_type == 'test':
            _, pred_probs, pred_labels, true_labels = self.evaluate(model, nn.CrossEntropyLoss(), self.test_dataloader)
        if loader_type == 'valid':
            _, pred_probs, pred_labels, true_labels = self.evaluate(model, nn.CrossEntropyLoss(), self.valid_dataloader)
        prec, recall, f1, auc_, acc = get_classification_reports(pred_probs, pred_labels, true_labels)
        return prec, recall, f1, auc_, acc, pred_probs, pred_labels, true_labels
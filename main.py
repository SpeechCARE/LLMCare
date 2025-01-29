from bidict import bidict
import itertools

All_transformers = bidict({'bert-base-uncased':'BERT', 'distilbert-base-uncased':'DistilBERT', 'roberta-base':'RoBERTa',
                    'distilroberta-base':'DistilRoBERTa', 'google/electra-base-discriminator':'ELECTRA',
                    'xlm-roberta-base':'XLM-R', 'xlnet-base-cased':'XLNet', 'allenai/longformer-base-4096':'Longformer'})

transformers_last_n_layers = {'bert-base-uncased':1, 'distilbert-base-uncased':1, 'roberta-base':1,
                             'distilroberta-base':1, 'google/electra-base-discriminator':1, 'xlm-roberta-base':1,
                              'xlnet-base-cased':1, 'allenai/longformer-base-4096':1}

transformers_list = ['distilbert-base-uncased'] #****
seeds_list = [0, 3, 10, 33, 55]
embedding_output_structure_list = ['CLS']

all_list = [transformers_list, seeds_list, embedding_output_structure_list]
episodes = list(itertools.product(*all_list))


def __main__():
    try:
        df_results = pd.read_excel(result_path + result_sheet)

        results = []
        for index, row in df_results.iterrows():
            results.append((All_transformers.inverse[row.Transformers], row.Seed, row.Embedding_output_structure))
        remaining_episodes = [e for e in episodes if e not in results]

    except FileNotFoundError:
        df_results = pd.DataFrame()
        remaining_episodes = episodes.copy()

    for row, episode in enumerate(remaining_episodes):

        configs['path_to_save_weights_based_f1'] = 'Saved_weights_based_maximum_f1.pt'
        configs['transformers_name'] = episode[0]
        configs['seed'] = episode[1]
        configs['embedding_output_structure'] = episode[2]
        configs['last_n_layers_list'] = transformers_last_n_layers[episode[0]]
        # Last n layers
        lnl = configs['last_n_layers_list']

        # training and validating
        print('\nTransformers: {} \n Seed: {} \n Embedding_output_structure: {} \n Last_n_layers: {}'.format(All_transformers[episode[0]], episode[1], episode[2], lnl))
        Atc_training = Alzheimer_text_classification(configs, train_data=train_data, valid_data=valid_data, phase='training')
        try:
            Atc_training.start_training_and_validating()
        except AttributeError:
            continue

        # testing
            valid_precision, valid_recall, valid_f1, valid_AUC, valid_Accuracy, valid_pred_probs, valid_pred_labels, valid_true_labels = Atc_training.testing(Atc_testing.path_max_f1, 'valid')
            test_precision, test_recall, test_f1, test_AUC, test_Accuracy, test_pred_probs, test_pred_labels, test_true_labels = Atc_testing.testing(Atc_testing.path_max_f1, 'test')
            print(f'Max F1 on Test: {test_f1}')
            print("-"*200)

            valid_new_probs1 = pd.DataFrame({'Seed':[episode[1]]*len(valid_true_labels),
                                    'Embedding_output_structure':[episode[2]]*len(valid_true_labels),
                                    'True_labels': valid_true_labels,
                                    'Pred_labels': valid_pred_labels,
                                    'Pred_probs_label_1' : valid_pred_probs})

            test_new_probs1 = pd.DataFrame({'Seed':[episode[1]]*len(test_true_labels),
                                    'Embedding_output_structure':[episode[2]]*len(test_true_labels),
                                    'True_labels': test_true_labels,
                                    'Pred_labels': test_pred_labels,
                                    'Pred_probs_label_1' : test_pred_probs})

            new_entry = pd.DataFrame({'Transformers':All_transformers[episode[0]],
                                    'Seed':episode[1],
                                    'Fine-tuning':configs['fine-tuning'] if configs['fine-tuning']!='Last_n_layers_of_transformers' else f'Last {lnl} layers' if lnl!=1 else f'Last {lnl} layer',
                                    'Embedding_output_structure':episode[2],
                                    'Max_sequence_length':configs['max_seq_len'],
                                    'Classification':'Two dense layers',
                                    'Batch_size':configs['batch_size'],
                                    'Epochs':configs['epochs'],
                                    'Learning_rate':configs['learning_rate'],
                                    'Data_source':'DePiC'+data_type,
                                    'Validation_type':'valid-test',
                                    'Best_epoch': Atc_training.best_epoch_f1,
                                    'valid_Precision':valid_precision,
                                    'valid_Recall':valid_recall,
                                    'valid_F1':valid_f1,
                                    'valid_AUC':valid_AUC,
                                    'valid_Accuracy':valid_Accuracy,
                                    'test_Precision':test_precision,
                                    'test_Recall':test_recall,
                                    'test_F1':test_f1,
                                    'test_AUC':test_AUC,
                                    'test_Accuracy':test_Accuracy,
                                    }, index=[0],)

            valid_df_probs1 = pd.concat([valid_df_probs1, valid_new_probs1], ignore_index=True)
            valid_df_probs1.to_excel(result_path + valid_predicted_probs_sheet, index=False)

            test_f_probs1 = pd.concat([test_df_probs1, test_new_probs1], ignore_index=True)
            test_df_probs1.to_excel(result_path + test_predicted_probs_sheet, index=False)

            df_results = pd.concat([df_results, new_entry], ignore_index=True)
            df_results.to_excel(result_path + result_sheet, index=False)

            print('#' * 40)
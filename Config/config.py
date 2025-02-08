TRANSCRIPTED = ''
ONCE_AUGMENTATION = '_once_augmentation'
TWICE_AUGMENTATION = '_twice_augmentation'

def get_config():
    data_type = TRANSCRIPTED

    configs = {}
    configs['max_seq_len'] = 512
    configs['batch_size'] = 8
    configs['epochs'] = 50
    configs['learning_rate'] = 2e-5
    configs['fine-tuning'] = 'Last_n_layers_of_transformers' # 'No fine-tuning' / 'All layers' / 'Last_n_layers_of_transformers'

    configs['extended'] = True
    configs['fusion_type'] = 'late3'
    configs['sub_inter_fusion'] = False
    configs['jmim'] = True

    configs['trans_mid_size'] = 256
    configs['ling_mid_size'] = 1024
    configs['liwc_mid_size'] = 128

    configs['data_type'] = data_type

    configs['ling_input_size'] = 32
    if configs['fusion_type'] == None:
        configs['liwc_input_size'] = 0
    elif data_type == TRANSCRIPTED:
        configs['liwc_input_size'] = 89 if not configs['jmim'] else 34
    elif data_type == ONCE_AUGMENTATION:
        configs['liwc_input_size'] = 90 if not configs['jmim'] else 57
    elif data_type == TWICE_AUGMENTATION:
        configs['liwc_input_size'] = 91 if not configs['jmim'] else 21

    return configs

    
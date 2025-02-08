from Config import *

configs = get_config()

MANUAL_LING = 'Manual'
LIWC = 'LIWC'



path = '/content/drive/MyDrive/Transformers_LLMs_Linguistic_Paper/Development/'
result_path = path + 'Result/Yasaman_DePiC/'
data_path = path + 'Data/'

Code_Name ='tansformer_auc_EarlyStopping_DePic'
result_sheet = f'Last_n_layers_of_{Code_Name}.xlsx'
valid_predicted_probs_sheet = f'/Validation_Predicted_probability_of_{Code_Name}.xlsx'
test_predicted_probs_sheet = f'/Test_Predicted_probability_of_{Code_Name}.xlsx'


train_data_path = data_path + 'New_Data_Yasaman/'
validation_data_path = data_path + 'New_Data_Yasaman/'
test_data_path = data_path + 'Test_data_2021/'

train_text_path = 'New_Data_Yasaman/Text/train.csv'
val_text_path = 'New_Data_Yasaman/Text/validation.csv'
test_text_path = "Test_data_2021/Text/Test_DePiC.xlsx"


# Linguistic features
train_ling_path = train_data_path + f"Linguistic/Linguistic_DePiC{configs['data_type']}_train.xlsx"
valid_ling_path = train_data_path + f"Linguistic/Linguistic_DePiC{configs['data_type']}_validation.xlsx"
test_ling_path =  test_data_path + "Linguistic/Test_Linguistic_DePiC.xlsx"


# LIWC features
train_LIWC_path = train_data_path + f"LIWC/LIWC_DePiC{configs['data_type']}_train.xlsx"
valid_LIWC_path = train_data_path + f"LIWC/LIWC_DePiC{configs['data_type']}_validation.xlsx"
test_LIWC_path =  test_data_path + "LIWC/Test_LIWC_DePiC.xlsx"

# JMIM LIWC
train_jmim_LIWC_path = train_data_path + f"LIWC/JMIM_LIWC_DePiC{configs['data_type']}_train.xlsx"
valid_jmim_LIWC_path = train_data_path + f"LIWC/JMIM_LIWC_DePiC{configs['data_type']}_validation.xlsx"
test_jmim_LIWC_path = test_data_path + f"LIWC/Test_JMIM_LIWC_DePiC{configs['data_type']}.xlsx"


import numpy as np
import pandas as pd
import os
metric_use = 'loss'
n_classes = 19
#WORK_LOCATION = f'data/submissions/test_ensamble_3/'


SCALING = 2.

WORK_LOCATION = f'data/submissions/test_{"v6_2_1"}_{metric_use}/'
test_enc_df = pd.read_csv(os.path.join(WORK_LOCATION,'stage_1.csv'))
WORK_LOCATION = f'data/submissions/test_{"v6_4"}_{metric_use}/'   #effb0 one
test_enc_df_1 = pd.read_csv(os.path.join(WORK_LOCATION,'stage_1.csv'))
WORK_LOCATION = f'data/submissions/test_{"v2_3_10_1"}_{metric_use}/'   #effb0 one
test_enc_df_2 = pd.read_csv(os.path.join(WORK_LOCATION,'stage_1.csv'))


WORK_LOCATION = f'data/submissions/test_ensamble_2/'

#print(test_enc_df.head())

#test_enc_df['15'] = np.clip(test_enc_df['15'].values * SCALING, 0.0, 1.0)
#test_enc_df['11'] = np.clip(test_enc_df['11'].values * SCALING, 0.0, 1.0)
#test_enc_df['17'] = np.clip(test_enc_df['17'].values * SCALING, 0.0, 1.0)

#test_enc_df['1'] = np.clip(test_enc_df['1'].values * SCALING, 0.0, 1.0)
#test_enc_df['9'] = np.clip(test_enc_df['9'].values * SCALING, 0.0, 1.0)
#test_enc_df['10'] = np.clip(test_enc_df['10'].values * SCALING, 0.0, 1.0)
#test_enc_df['6'] = np.clip(test_enc_df['6'].values * SCALING, 0.0, 1.0)
#test_enc_df['11'] = np.clip(test_enc_df['11'].values * SCALING, 0.0, 1.0)
#test_enc_df['11'] = np.clip(test_enc_df['11'].values * SCALING, 0.0, 1.0)


#print('After scaling ')

#print(test_enc_df.head())

#we will see how the ensamble will work
predictions = test_enc_df[[str(i) for i in range(n_classes)]].values + test_enc_df_1[[str(i) for i in range(n_classes)]].values + test_enc_df_2[[str(i) for i in range(n_classes)]].values
test_enc_df[[str(i) for i in range(n_classes)]] = predictions/3.

#test_enc_df['0'] = np.clip(test_enc_df['0'].values * SCALING, 0.0, 1.0)
#test_enc_df['16'] = np.clip(test_enc_df['16'].values * SCALING, 0.0, 1.0)

tokens_list = test_enc_df.ID.unique()

prediction_string_list = []
token_list = []
for tok in tokens_list:
    prediction_str = ''
    sub_d = test_enc_df[test_enc_df['ID'] == tok]
    for i in range(len(sub_d)):
        info = sub_d.iloc[i]
        encoding = info['encoding']
        class_pred = info[[str(j) for j in range(n_classes)]].values
        for count, k in enumerate(class_pred):
            prediction_str += f'{count} {k} ' + encoding + ' '
    #here we might have to check if the string has len > 0 . maybe we might get '' also ......
    prediction_str = prediction_str.strip()# hopefuly removes the final space
    token_list.append(tok)  
    prediction_string_list.append(prediction_str)

sub_stage_2_df = pd.DataFrame.from_dict({'ID':token_list,"PredictionString":prediction_string_list })

sub = pd.read_csv('data/sample_submission.csv')
sub = sub.drop(['PredictionString'],axis=1)
sub = sub.merge(sub_stage_2_df, on='ID')

WORK_LOCATION = f'data/submissions/test_ensamble_3/'

if not os.path.exists(WORK_LOCATION):
        os.mkdir(WORK_LOCATION)
sub.to_csv(os.path.join(WORK_LOCATION,'submission_ensamble_3.csv'), index=False)
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import itertools
from copy import deepcopy

sns.set()

# lay out the data, taken from BERT code results. all data to go in df
acc = [92, 94, 94, 95, 95, 93]
prec = [94, 94, 94, 96, 96, 96]
rec = [95, 98, 98, 98, 98, 95]
f1 = [94, 96, 96, 97, 97, 96]
#scores = acc + prec + rec + f1
score_dict ={'Accuracy':acc,'Precision':prec,'Recall':rec,'F1':f1}

rf_acc = [81, 81, 83, 85, 88,90]
rf_prec = [82, 82, 84, 84, 90,92]
rf_rec = [96, 96, 95, 97, 94,94]
rf_f1 = [88, 88, 89, 90, 92,93]
#rf_scores = rf_acc + rf_prec + rf_rec + rf_f1
rf_score_dict = {'Accuracy':rf_acc,'Precision':rf_prec,'Recall':rf_rec,'F1':rf_f1}

# training scores - testing scores
acc_dif = [4, 3, 3, 2, 3, 4]
prec_dif = [4, 4, 3, 2, 2, 2]
rec_dif = [2, 1, 1, 0.2, 1, 3]
f1_dif = [3, 2, 2, 1, 2, 2]
#scores_dif = acc_dif + prec_dif + rec_dif + f1_dif
score_dif_dict ={'Accuracy':acc_dif,'Precision':prec_dif,'Recall':rec_dif,'F1':f1_dif}

rf_acc_dif = [0.2,6,4,0.2,-1,-1]
rf_prec_dif = [0.2,5,2,2,0.2,0.2]
rf_rec_dif = [1,4,2,0.2,-1,0.2]
rf_f1_dif = [0.2,1,3,0.2,0.2,0.2]
rf_score_dif_dict={'Accuracy':rf_acc_dif,'Precision':rf_prec_dif,'Recall':rf_rec_dif,'F1':rf_f1_dif}
# construct df
radii = [1,2,3,5,10, "None"]
models = ['BERT', 'RF']
radii_lp = [1,2,3,4,5,6]
set_type = ['test', 'train']
score_types = ['Accuracy', 'Precision', 'Recall', 'F1']
tuples = list(itertools.product(models, radii))
# for score_type in score_types:
    
#     df=pd.DataFrame(columns=['radius','set_type'])
#     df['radius'] = [pair[1] for pair in tuples]
#     df['set_type'] = [pair[0] for pair in tuples]
#     scores = score_dict[score_type]
#     score_difs = score_dif_dict[score_type]
#     score_vals = scores+[scores[idx]+score_difs[idx] for idx in range(len(scores))]
#     df['score'] = score_vals
    
#     ax = sns.barplot(x='radius', y='score',hue='set_type',data=df)
#     rf_scores = rf_score_dict[score_type]
#     rf_med_score = np.median(rf_scores)
#     ax.plot(radii,[rf_med_score]*6,'g--')
#     ax.set_ylim(80,100)
#     ax.set_title(score_type+' Scores',fontsize='14')
#     fig = ax.get_figure()
#     fig.savefig(score_type+'_RF_median_baseline.png',dpi=200)
#     fig.clf()

# df = pd.DataFrame(columns=['radius', 'score_type'])
# df['radius'] = [pair[1] for pair in tuples]
# df['score_type'] = [pair[0] for pair in tuples]
# df['score'] = scores
# ax = sns.lineplot(x='radius', y='score', hue='score_type',data=df)
#rf_scores = rf_score_dict[score_type]
#rf_med_score = np.median(rf_scores)
#ax.plot(radii,[rf_med_score]*6,'g--')
# ax.set_xticklabels(radii)
# ax.set_ylim(90,100)
# ax.set_title('Scores Distribution', fontsize=14)
# fig = ax.get_figure()
# fig.savefig('lineplot.png',dpi=200)

for score_type in score_types:
    bert_scores = score_dif_dict[score_type]
    rf_scores = rf_score_dif_dict[score_type]
    
    df = pd.DataFrame(columns=['model','radius'])
    df['model'] = [tup[0] for tup in tuples]
    df['radius'] = [tup[1] for tup in tuples]
    df_dif = deepcopy(df)
    df['difference'] = bert_scores+rf_scores
    # df_dif['score_difference'] = scores_dif
    
    ### plot
    #col_pal = sns.color_palette('Blues_d')[::-1]
    ax = sns.barplot(x="radius", y="difference", hue="model",
                      data=df)#, palette=col_pal)
    ax.set_ylim(-2,8)
    ax.set_title(f'Training set score - Testing set score ({score_type})')
    fig = ax.get_figure()
    fig.savefig(score_type+'_differences_barplot.png',dpi=200)
    fig.clf()

# ax_dif = sns.barplot(x='type', y='score_difference', hue='radius',
#                       data=df_dif, palette='Reds')
# ax_dif.set_title('Difference in Training Scores Vs. Testing Scores')
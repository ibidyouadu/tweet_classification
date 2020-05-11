import utils
import numpy as np
import pandas as pd
# from copy import deepcopy

# addr='home/adu/Documents/research/twitter and disaster damage assessment'
# twt = pd.read_csv(addr+'/code/data/harvey_tweets.csv')
# twt = twt[twt.state=='Texas']
# clean_twt = deepcopy(twt)
# clean_twt.text = clean_twt.text.apply(utils.tweet_clean)


# kwds = {}
# pubhealth = 'pubhealth'
# housing = 'housing'
# transport = 'transport'
# power = 'power'
# kwds[pubhealth] = 'hospital hospitalize hospitalise injure injury'.split() \
#     + 'casualty fatality dead die rescue miss evacuate'.split()
# kwds[housing] = 'home house tree roof shelter retirement'.split()
# kwds[transport] = 'accident close block street road lane highway'.split() \
#     + 'rd ln hwy freeway fwy tollway twy parkway pwy loop lp 45 i45'.split() \
#     + '10 i10 i69 69 i610 610 beltway8 belt8 airport'.split()
# kwds[power] = 'outage blackout electricity power'.split()
# ^^^ POWER KWDS FOR IRMA ONLY
# job_idx = utils.keyword_filtering(twt.text, ['job', 'hire'])
# idx = twt.index

# recipe:
# pubhealth_idx = idx[utils.keyword_filtering(clean_twt.text, kwds[pubhealth])]
# pubhealth_idx = [x for x in pubhealth_idx if x not in job_idx]
# pubhealth_twt = twt.filter(items=pubhealth_idx, axis=0)
# then read through tweets!

def kwd_sample(twt,kwd,score):
    # create list of idx of sample tweets carrying kwd. len(list) <= 20
    # also returns number of tweets filtered using kwd
    kwd_idx = utils.keyword_filtering(twt.text, [kwd])
    filter_size = len(kwd_idx) # total tweets returned
    if kwd == 'power':
        loop_size = 1526
    else:
        loop_size = filter_size # number of tweets to sample
    sample_idx = []
    # if filter_size >= 20:
    #     loop_size = 20
    # else:
    #     loop_size = filter_size
    for n in range(loop_size):
        new_idx = kwd_idx[np.random.randint(0,filter_size)]
        sample_idx.append(new_idx)
        kwd_idx = kwd_idx[kwd_idx != new_idx]
        filter_size -= 1    
    return sample_idx, filter_size

def generate_samples(ctwt, twt, kwds,scores, key):
    # write file containing text of sample tweets for all kwds
    # with open(fname +'.txt', 'w') as f:
    #     # for key in kwds:
    #         # f.write('CATEGORY: ' + key.upper() + '\n\n')
    #     for kwd in kwds[key]:
    #         kwd_idx, kwd_num_twt = kwd_sample(ctwt, kwd, scores[kwd])
    #         kwd_twt = twt.filter(items=kwd_idx,axis=0)
    #         kwd_txt = kwd_twt.text.values
    #         f.write(kwd.upper() + ' TWEETS (returned ' + str(kwd_num_twt) + ' tweets)\n')
            
    #         for idx in range(len(kwd_txt)):
    #             f.write(str(idx+1) + '. ' + kwd_txt[idx] + '\n\n')
    #         f.write('\n')
    #     f.write('\n\n\n')
    for kwd in kwds[key]:
        kwd_idx, kwd_num_twt = kwd_sample(ctwt, kwd, scores[kwd])
        kwd_df = twt.filter(items=kwd_idx, axis=0)
        kwd_df['topic_related'] = 0
        kwd_df['dis_related'] = 0
        kwd_df['dmg_related'] = 0
        kwd_df['sentiment'] = np.nan
        kwd_df.to_csv(kwd+'.csv',index=False)
        
def record_results(kwds):
    # kwds is the list of kwds, not dict
    # run this in directing containing csv's of keyword sample tweets
    res = pd.DataFrame(columns='kwd num_topic_related num_dis_related num_dmg_related neg_sentiment neut_sentiment pos_sentiment'.split())
    res['kwd'] = kwds
    n_top = []
    n_dis = []
    n_dmg = []
    neg_sent = []
    neut_sent = []
    pos_sent = []
    
    for kwd in kwds:
        df = pd.read_csv(kwd + '.csv') # make sure csv's are in current directory!!
        n_top.append(np.sum(df.topic_related.values))
        n_dis.append(np.sum(df.dis_related.values))
        n_dmg.append(np.sum(df.dmg_related.values))
        n_sent = [0,0,0] # sentiment is an R^3 vector. example: [1,2,3]
        # means 1 tweet negative, 2 tweets neutral, 3 tweets positive
        for s in df.query('dis_related==1 | topic_related==1').sentiment.values:
            if s==-1:
                n_sent[0] += 1
            elif s==0:
                n_sent[1] += 1
            elif s==1:
                n_sent[2] += 1
        neg_sent.append(n_sent[0])
        neut_sent.append(n_sent[1])
        pos_sent.append(n_sent[2])
    res['num_topic_related'] = n_top
    res['num_dis_related'] = n_dis
    res['num_dmg_related'] = n_dmg
    res['neg_sentiment'] = neg_sent
    res['neut_sentiment'] = neut_sent
    res['pos_sentiment'] = pos_sent
    res.to_csv('results.csv',index=False)
    
    
########## Thurs March 05
# twt = pd.read_csv('irma_florida_tweets_job_filter.csv')
# counties = twt.county.unique()
# counties.sort()
# counties = np.array([c[:-7] for c in counties]) #chop off the ' County' in strings
# dates = twt.created_at.unique()
# t = pd.to_datetime(dates)
# t = t.sort_values()

# topic = pd.DataFrame(columns=np.append(['date'],counties))
# topic.set_index('date', inplace=True)
# pos_sent = deepcopy(topic)
# neut_sent = deepcopy(topic)
# neg_sent = deepcopy(topic)
# kwd_df = pd.DataFrame()
# for kwd in kwds:
#     kwd_df = pd.read_csv(kwd+'.csv')
# for c in counties:
#     topic_vals = np.array([])
#     pos_vals = deepcopy(topic_vals)
#     neut_vals = deepcopy(topic_vals)
#     neg_vals = deepcopy(topic_vals)
#     c_df = kwd_df[kwd_df.county==c+' County']
#     for d in t: #important! make sure t is sorted, otherwise the plots are messed upper
#         d_df = c_df[pd.to_datetime(c_df.created_at)==d]
#         num_topic = d_df[d_df.topic_related==1].shape[0]
#         num_pos = d_df[d_df.sentiment==1].shape[0]
#         num_neut = d_df[d_df.sentiment==0].shape[0]
#         num_neg = d_df[d_df.sentiment==-1].shape[0]
#         topic_vals = np.append(topic_vals, num_topic)
#         pos_vals = np.append(pos_vals, num_pos)
#         neut_vals = np.append(neut_vals, num_neut)
#         neg_vals = np.append(neg_vals, num_neg)
#     topic[c] = topic_vals
#     pos_sent[c] = pos_vals
#     neut_sent[c] = neut_vals
#     neg_sent[c] = neg_vals

# for c in counties:
#     make_plot(t, topic[c], 'Topic-related Tweets', c+' County Topic-related\nTweets Over Time', c+'.png')
# # etc for sentiment plots
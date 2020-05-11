import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from copy import deepcopy
import random

def clean_outage_data():
    # transform original dataset into nominal and 'per-capita' time series
    # note that the original dataset consists of 'customers' which does not 
    # mean 1 person necessarily. the 'per-capita' timeseries is normalized by customers,
    # hence why I use the term 'per-capita' lightly
    
    pwr = pd.read_excel('outages_original.xlsx')
    pwr = pwr.rename({'County':'county', 'PowerProvider':'provider', \
                'Customers':'customers', 'Out': 'out', 'CreatedOn': 'date'},axis=1)
    pwr.county = pwr.county.apply(lambda x: ' '.join([w.capitalize() for w in x.split()]))
    
    def rename_counties(x):
        # this is really just to fix the capitalization on Miami-Dade
        if x=='Miami-dade':
            return 'Miami-Dade'
        else:
            return x
        
    pwr.county = pwr.county.apply(rename_counties)
    counties = pwr.county.unique()
    
    # the following is for date formatting so that dates.sort() works properly
    pwr.date = pwr.date.apply(lambda x: np.datetime_as_string(np.datetime64(x))[:10])
    dates = pwr.date.unique()
    dates.sort()
    
    # the following will be converted to dataframes. they are the time series
    out_dict = {}
    out_percent_dict = {}
    
    for county in counties:
        county_customers = np.array([]) # customers for each date
        county_out = np.array([]) # affected customers for each date
        countyDF = pwr[pwr.county==county]
        providers = countyDF.provider.unique()
        
        county_customers_total = 0 # total number of customers between all
        # providers of the county. We calculate it like so: go through each
        # day, and each provider. Average the customer count for the provider's
        # counts (they can have multiple readings in a day). Add up the averages
        # of each provider. That's the count for the day. Finally, take the
        # maximum count from the whole month.
        # The reason we use this number will be clear later
        
        # the following loop finds county_customers_total
        # We have two separate loops across the dates because we want to know 
        # this number first. It does look awkward, don't hesitate to modify it
        for date in dates:
            count = 0 # the customers we count for the given date
            today_countyDF = countyDF[countyDF.date==date]
            for prov in providers:
                    today_prov_countyDF = today_countyDF[today_countyDF.provider==prov]
                    count += round(np.nanmean(today_prov_countyDF.customers))
            county_customers_total = max(county_customers_total, count)
        
        for date in dates:
            date_customers = 0
            date_out = 0
            for prov in providers:
                # again, a given provider on a given day for the given county
                # can have multiple readings (rows). So we take averages
                rows = countyDF[countyDF.date==date][countyDF.provider==prov]
                if rows.shape[0] > 0:
                    date_customers += round(np.nanmean(rows.customers.values))
                    date_out += round(np.nanmean(rows.out.values))
            
            # The following is why we needed county_customers_total
            # There are some cases where date_customers is very small
            # This causes the 'per-capita' time series to have outliers
            # So if the customers counted for on a given day are not enough
            # (for what ever reason -- missing data, readings that weren't
            # recorded, etc.) we simply don't write anyything for that day
            # the threshold of 90% of county_customer_total is somewhat arbitrary
            # I haven't tried fiddling with it, but if there comes any time to
            # review how the timeseries were produced, this may be something
            # to look at
                    
            if date_customers < 0.9*county_customers_total:
                date_customers = np.nan
                date_out = np.nan
                
            # record numbers, compute 'per-capita' numbers
            county_customers = np.append(county_customers, date_customers)
            county_out = np.append(county_out, date_out)
            county_out_percent = county_out / county_customers
            out_percent_dict.update({county: county_out_percent})
            out_dict.update({county: county_out})
    
    pwr_ts = pd.DataFrame(out_dict, index=dates).reset_index().rename({'index':'date'},axis=1)
    pwr_percent_ts = pd.DataFrame(out_percent_dict, index=dates).reset_index().rename({'index':'date'},axis=1)

    return pwr_ts, pwr_percent_ts

def impact_ratio(twt, c, pp_date):
    pp_idx = twt[twt.date==pp_date].index.values[0]
    before = twt.iloc[pp_idx-1:pp_idx][c]
    beforesum = np.nansum(before.values)
    after = twt.iloc[pp_idx:pp_idx+1][c]
    aftersum = np.nansum(after.values)
    ir_out = beforesum/aftersum
    return ir_out

def county_active_filter(twtDF,tol):
    out = np.array([])
    for c in twtDF.county.values:
        if twtDF[twtDF.county==c].total.values[0] >= tol:
            out = np.append(out, c)
    
    return out

def corr_iter(df1, col1, df2, col2):
    out = {} # {tol: pearsonr corr}
    tols = deepcopy(df1[col1].unique()) #cut off counties using values from df1[col1]
    tols.sort()
       
    for tol in tols[:-1]:
        active = df1[df1[col1]>=tol].county.values
        df1_act = df1[df1.county.isin(active)]
        df2_act = df2[df2.county.isin(active)]
        corr = pearsonr(df1[col1], df2[col2])
        out.update({tol: corr})
    return out

def DTWD(t1, t2):
    DTW = {}
    idxs = range(len(t1)) # assuming t1, t2 equal length
    
    # initialize norms
    for i in idxs:
        DTW[(i,-1)] = float('inf')
        DTW[(-1,i)] = float('inf')
    DTW[(-1,-1)] = 0
    
    for i in idxs:
        for j in idxs:
            dist = (t1[i] - t2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i, j-1)], DTW[(i-1, j)], DTW[(i-1, j-1)])
    
    return np.sqrt(DTW[idxs[-1], idxs[-1]])


# ntwt = pd.read_csv('daily_activity_normalized.csv')
# X = ntwt[test_counties].transpose().values
def kmeans_DTWD(data, num_clust, num_iter, do_random, centroids=[]):
    if do_random==1:
        centroids = random.sample(data.tolist(), num_clust)
    if centroids == []:
        print("Don't forget to include the initial centroids!")
        return True
    
    for n in range(num_iter):
        labels = {}
        for idxi, ti in enumerate(data):
            nearest_clust = 0
            nearest_dist = DTWD(ti, centroids[0])
            
            for idxj, tj in enumerate(centroids):
                if DTWD(ti, tj) < nearest_dist:
                    nearest_dist = DTWD(ti, tj)
                    nearest_clust = idxj
            
            if nearest_clust not in labels:
                labels[nearest_clust] = []
            labels[nearest_clust].append(idxi)
        
        for label in labels:
            centroids[label] = np.sum(data[labels[label]], axis=0) / len(labels[label])
    return labels
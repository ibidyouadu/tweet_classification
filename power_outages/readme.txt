power_outages is the code used to produce outages_daily.csv and outages_daily_normalized.csv from outages_original.xlsx

There are also some functions to produce metrics and for finding patterns in the data using dynamic time warping and kmeans, mostly copied from some medium article. This was fruitless and can be ignored for the most part.

It is important to understand the data here: all the numbers in outages_daily.csv and outages_original.xlsx represent customers, which are not necessarily individual people. I tried finding more information about the customers, particularly the square footage of the properties, in order to get a hint of exactly how many people would a given customer consist of. I consulted the original source of the outage data, Florida Division of Emergency Management, but to no avail.

Similarly, outages_daily_normalized.csv has customers without power normalized by customers in total. The customers in total bit is a little complicated. See the code and comments to see what I mean.
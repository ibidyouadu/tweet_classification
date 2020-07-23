# BERT Tweet Classifier with Keyword Neighbors

## In a nutshell
This contains all data, code, results, etc. for a tweet classification project as a part of my work with [Dr. Wenying Ji's civil engineering research group at George Mason University.](http://mason.gmu.edu/~wji2/team.xhtml)

This project tackles the broad question *Can we use social media data (tweets) to assess the severity of natural disasters?* and more specifically addresses the question *Will using context surrounding keywords in tweets provide better results for text classifiers to label tweets as related to power infrastructure or not?*


## Example of what the code does
More concretely, let's say we have the following tweet:

`Throwback to when my apartment had electricity?????? #ThanksIrma @ Orlando, Florida https://t.co/R9YnR8sR8y`

We would like to automatically label it as related to power infrastructure or not. In this case, it is related since it talks about electric utility service. Before we feed it to a text classifier (which in this project, is either a BERT keras model or a random forest) we want to clean the text and slice the text so that our string slice is centered around a certain keyword  and within a specified "distance" i.e. number of words. We call these string slices **neighborhoods** about our keyword, which in this case is `electricity`. After cleaning the text and using a neighborhood radius of 2, we get the following:

`throwback to when my apartment had electricity?????? #ThanksIrma @ Orlando,`

Even though the radius is 2, there are more than 2 words to the left of the keyword because a certain class of words (stop words) are not counted. It is this string that is then input into a text classifier. Our results indicate that performance for the BERT classifer improved, whereas the random forest classifier did not.

## Code and Documentation
That is the project in a nutshell. The random forest model code is [here](https://github.com/ibidyouadu/tweet_classification/blob/master/modeling/RF_tweet_classification.ipynb) and the BERT model code is [here](https://github.com/ibidyouadu/tweet_classification/blob/master/modeling/BERT_tweet_classification.ipynb), in the form of annotated jupyter notebooks. For more details on the research methods and results, you can find the documentation [here](https://github.com/ibidyouadu/tweet_classification/blob/master/documentation/tweet_classification_documentation.pdf).

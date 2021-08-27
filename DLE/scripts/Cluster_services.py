#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from configparser import ConfigParser
#Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")
#Get the _PATH_ROOT_
_PATH_ROOT_ = config_object["Path"]["root"]


import sys
sys.path.insert(1, _PATH_ROOT_ +'/DLE/utils')
sys.path.insert(1, _PATH_ROOT_ +'/DLE/models')
from ImportingModules import *
from ServiceClustering import *
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import LabelEncoder


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

def plot_wordcloud(df: pd.DataFrame, category: str, target: int) -> None:
    words = " ".join(df[df["Clusters"] == category]["Description"].values)
    plt.rcParams['figure.figsize'] = 10, 20
    wordcloud = WordCloud(stopwords=STOPWORDS,background_color="white",max_words=1000).generate(words)
    plt.title("WordCloud For {}".format(category))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def plot_barchart(df: pd.DataFrame, clmn: str, title='Bar Chart') -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fg, axs = plt.subplots(1, 1, figsize=(15, 20))
    ax = sns.countplot(y=clmn, data=df)
    plt.title(title)


######Cluster Services#########

serviceClustertObj = ServiceCluster()
print("loadData ....")
df = serviceClustertObj.loadData()
print("PreProcessing started....")
start = time.time()
df3 = serviceClustertObj.preProcessLabels("Category")
serviceClustertObj.filterLowFrequencyCategory(10)
serviceClustertObj.removeNullrows("Category")
serviceClustertObj.removeNullrows("Description")
# serviceClustertObj.preprocessServicesData("Description")
end = time.time()
print("PreProcessing completed....")
print(end - start)

# clustering
serviceClustertObj.preprocessServicesData("Description")
df4 = serviceClustertObj.clusterKMeansUsingBert("Category")
# df4 =df4.rename(columns = {'clusters': 'Clusters'}, inplace = False)

#Investigate clustered dtaset
# df_grouped=df4.groupby(['clusters'], as_index=False).agg({ 'Category':lambda x:",".join(map(str, x))})
# df_grouped['Category']  = df_grouped['Category'].apply(lambda x:' '.join(unique_list(x.split(','))))
# df_grouped

# Providing dataset
df4 = df4.rename(columns={'clusters': 'Clusters'}, inplace=False)
df4 = df4.drop(['target'], errors='ignore', axis=1)
df_final = df4.drop(['Unnamed: 0.1'], errors='ignore', axis=1)
df_final['Category'] = ''
df_final.loc[df_final.Clusters == 0, ['Category']] = 'Financial'
df_final.loc[df_final.Clusters == 1, ['Category']] = 'Photos'
df_final.loc[df_final.Clusters == 2, ['Category']] = 'Storage'
df_final.loc[df_final.Clusters == 3, ['Category']] = 'Messaging'
df_final.loc[df_final.Clusters == 4, ['Category']] = 'Tools'  #
df_final.loc[df_final.Clusters == 5, ['Category']] = 'Commerce'
df_final.loc[df_final.Clusters == 6, ['Category']] = 'Sports'
df_final.loc[df_final.Clusters == 7, ['Category']] = 'Security'
df_final.loc[df_final.Clusters == 8, ['Category']] = 'Science'
df_final.loc[df_final.Clusters == 9, ['Category']] = 'Enterprise'
df_final.loc[df_final.Clusters == 10, ['Category']] = 'Social Media'
df_final.loc[df_final.Clusters == 11, ['Category']] = 'Text utils/Translation'
df_final.loc[df_final.Clusters == 12, ['Category']] = 'Mapping/Geography'
df_final.loc[df_final.Clusters == 13, ['Category']] = 'Payments'
df_final.loc[df_final.Clusters == 14, ['Category']] = 'Transportation'
df_final.loc[df_final.Clusters == 15, ['Category']] = 'Advertising/Marketing/Sales'
df_final.loc[df_final.Clusters == 16, ['Category']] = 'Government'
df_final.loc[df_final.Clusters == 17, ['Category']] = 'Telephony/Phone'
df_final.loc[df_final.Clusters == 18, ['Category']] = 'Internet'
df_final.loc[df_final.Clusters == 19, ['Category']] = 'Dashboards'
df_final.loc[df_final.Clusters == 20, ['Category']] = 'Content Feeds Semantics COVID Hosting Rewards'  ###
df_final.loc[df_final.Clusters == 21, ['Category']] = 'Artificial Intelligence/Analytics'
df_final.loc[df_final.Clusters == 22, ['Category']] = 'Education'
df_final.loc[df_final.Clusters == 23, ['Category']] = 'Cloud Computing'
df_final.loc[df_final.Clusters == 24, ['Category']] = 'Health'
df_final.loc[df_final.Clusters == 25, ['Category']] = 'Cryptocurrency'
df_final.loc[df_final.Clusters == 26, ['Category']] = 'Forms Applications Application Mobile Optimiza'  ##
df_final.loc[df_final.Clusters == 27, ['Category']] = 'Video/Movies'
df_final.loc[df_final.Clusters == 28, ['Category']] = ''
df_final.loc[df_final.Clusters == 29, ['Category']] = 'Weather'
df_final.loc[df_final.Clusters == 30, ['Category']] = 'Banking'
df_final.loc[df_final.Clusters == 31, ['Category']] = 'Stocks'
df_final.loc[df_final.Clusters == 32, ['Category']] = 'Search'  #
df_final.loc[df_final.Clusters == 33, ['Category']] = 'Email'
df_final.loc[df_final.Clusters == 34, ['Category']] = 'Music/Audio'
df_final.loc[df_final.Clusters == 35, ['Category']] = 'Media News Reporting'
df_final.loc[df_final.Clusters == 36, ['Category']] = 'Travel'
df_final.loc[df_final.Clusters == 37, ['Category']] = 'Bitcoin'
df_final.loc[df_final.Clusters == 38, ['Category']] = 'Monitoring/esting /Validation'
df_final.loc[df_final.Clusters == 39, ['Category']] = 'Shipping'
df_final.loc[df_final.Clusters == 40, ['Category']] = 'Processing'
df_final.loc[df_final.Clusters == 41, ['Category']] = 'Chat/Bots'
df_final.loc[df_final.Clusters == 42, ['Category']] = 'Backend'
df_final.loc[df_final.Clusters == 43, ['Category']] = 'Jobs/Office/Management'
df_final.loc[df_final.Clusters == 44, ['Category']] = 'Database'
df_final.loc[df_final.Clusters == 45, ['Category']] = 'PDF/Text Documents'
df_final.loc[df_final.Clusters == 46, ['Category']] = 'Project Managment'
df_final.loc[df_final.Clusters == 47, ['Category']] = 'Games'
df_final.loc[df_final.Clusters == 48, ['Category']] = 'Entertainment/Hotels'
df_final.loc[df_final.Clusters == 49, ['Category']] = 'TBot/Machine Learning'

# indexNames =df_final[
#     (df_final['Clusters'] == 32)
#     | (df_final['Clusters'] == 20)
#     | (df_final['Clusters'] == 26)].index
# df_final.drop(indexNames , inplace=True)
# Save
from datetime import datetime
# df_final.to_csv('Clustered_services_using_bert_'+datetime.now().strftime("%Y-%m-%d_%H:%M"))


# Plot word Cloud
labeler = LabelEncoder()
df4["target"] = labeler.fit_transform(df4["Clusters"])
mapping = []
for i in range(0, len(labeler.classes_)):
    mapping.append(labeler.classes_[i])

for i in range(0, len(mapping)):
    plot_wordcloud(df4, mapping[i], target=0)
#     mapping.append(labeler.classes_[i])


# Plot barchart
plot_barchart(df_final, "Category")

# Plot missed values
fg, axs = plt.subplots(1, 1, figsize=(10, 7))
sns.heatmap(df_final.isnull(), cbar=False)
print(df_final.info())
df_final.describe()
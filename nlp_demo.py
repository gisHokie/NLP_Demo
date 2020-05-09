# Used Miniconda to test
# Require NLTK, GEOPY, Scikit-Learn, Numpy, and CSV imports
# This is still in raw draft, which means minimal comments and explanations

#from stanza.server import CoreNLPClient
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *

from nltk import Tree
from nltk import download
from nltk import word_tokenize
from nltk import pos_tag
from nltk.sem import *

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np

import pandas as pd

# may need to use conda to install cartopy: conda install -c conda-forge cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import statistics 
from statistics import mode 



import csv
import os

'''RESOURCES
Book: https://www.nltk.org/book/
https://colab.research.google.com/github/stanfordnlp/stanza/blob/master/demo/Stanza_Beginners_Guide.ipynb
https://colab.research.google.com/github/stanfordnlp/stanza/blob/master/demo/Stanza_CoreNLP_Interface.ipynb#scrollTo=xrJq8lZ3Nw7b
https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/
nltk package: http://www.nltk.org/api/nltk.html
nltk.download() https://stackoverflow.com/questions/22211525/how-do-i-download-nltk-data
List of part of speech tags: https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
'''

# VARIABLES
csv_states = '/home/scotty/Projects/NLP/csv/us_states.csv'
csv_world = '/home/scotty/Projects/NLP/csv/worldcities.csv'
lst_states = []
lst_popular_cities = []

# NLTK download
#nltk.download('conll2000')
# nltk.download('popular') # download the popular model
download('punkt')
download('averaged_perceptron_tagger')
download('maxent_ne_chunker')
download('words')

# Annotate some text
text = """I live in Richmond,VA near the state capital and a lot of food.  
I live near Philip Morris Factory.  I have family that lives in Chicago, IL, Dallas, TX, Houston, Texas
Chesterfield, Ameilia, and Powhatan Virgina.
my favorite place in Virginia is Virginia Tech, Blacksburg, Va, 
starbucks for coffee,
and Buffalo Wild Wings for beer and wings and finally Can Can because my wife loves that place
"""

### CORENLP ###
# Set the CORENLP_HOME environment variable to point to the installation location
#locat_nlp = '/home/scotty/Projects/NLP/corenlp'
#os.environ["CORENLP_HOME"] = locat_nlp
# "/home/scotty/Projects/NLP/stanford-corenlp/"

# Construct a CoreNLPClient with some basic annotators, a memory allocation of 4GB, and port number 9001
#client = CoreNLPClient(annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner'], memory='4G', endpoint='http://localhost:9000')
# print(client)

#document = client.annotate(text)
#print(document)
# Iterate over all tokens in all sentences, and print out the word, lemma, pos and ner tags
# print("{:12s}\t{:12s}\t{:6s}\t{}".format("Word", "Lemma", "POS", "NER"))
#print(document.sentence)
#for i, sent in enumerate(document.sentence):
    #print("[Sentence {}]".format(i+1))
    #print(i)
    #for t in sent.token:
        #print("{:12s}\t{:12s}\t{:6s}\t{}".format(t.word, t.lemma, t.pos, t.ner))
        #print(t)
### CORENLP ###

#### CHUNK EXAMPLE ####
# http://www.nltk.org/howto/chunk.html
grammar = ChunkRule("<DT>?<JJ>*<NN>", "Chunk Grammar")
chunk_rule = ChunkRule("<.*>+", "Chunk everything")
chink_rule = ChinkRule("<VBD|IN|\.>", "Chink on verbs/prepositions")
split_rule = SplitRule("<DT><NN>", "<DT><NN>", "Split successive determiner/noun pairs")

# test nltk 
token_text = word_tokenize(text)
#print(token_text)
tagged_text = pos_tag(token_text)
#print(tagged_text)
chunk_tag = ne_chunk(tagged_text)
#print(chunk_tag)

# Grow a tree
#https://www.nltk.org/book/ch07.html
chunk_parser = RegexpChunkParser([chunk_rule, chink_rule, split_rule], chunk_label='NP')
chunked_text = chunk_parser.parse(chunk_tag)
#print('CHUNKY')
#print(chunked_text)

#chunkstr_text = ChunkString(chunked_text)
#print(chunkstr_text)

### Draw the Chunked Text Diagram  ###
# chunked_text.draw()
### print leaves
#print(chunked_text.leaves())

### Get the toponyms and other places from chunk ###
# https://stackoverflow.com/questions/48660547/how-can-i-extract-gpelocation-using-nltk-ne-chunk
# https://stackoverflow.com/questions/24398536/named-entity-recognition-with-regular-expression-nltk
continuous_chunk = []
current_chunk = []
labels = ['GPE', 'PERSON', 'ORGANIZATION']
for subtree in chunk_tag:
    if type(subtree) == Tree and subtree.label() in labels:
        current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
    elif current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)
            current_chunk = []
    else:
        continue
#print(continuous_chunk)

### Geocode the Places ###
# First Identify all city and states (ignore the non-administrative places for now)
# cluster the locations
# the cluster with the most items wins and should be the primary location of the text

# Locate the state
# Reads from CSV specific for USA States as faster than reading a map or gazetteer
tmp_chunk_list = continuous_chunk
for cc in continuous_chunk:
    with open(csv_states) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')        
        for row in reader:
            # convert item in list to upper case
            lst_upper = [x.upper() for x in row] 
            if cc.upper() in lst_upper: 
                lst_states.append(cc.upper())
    # Lets check if there are any cities that might be considered popular or significant in size or purpose
    # At this point we will assume the cities can be anywhere in the world
    with open(csv_world, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # convert item in list to upper case
            lst_upper = [x.upper() for x in row]
            if cc.upper() in lst_upper and cc.upper() not in lst_states:
                tmp_list = [lst_upper[1], lst_upper[7], lst_upper[4]]
                lst_popular_cities.append(tmp_list)

#LEFTOVERS
chunk_upper = [x.upper() for x in continuous_chunk]
tmp_chunk_upper = [x.upper() for x in tmp_chunk_list]
lst_upper_states = [x.upper() for x in lst_states] 

# Remove the states from main chunk list
for tmp in chunk_upper:
    if tmp.upper() in lst_upper_states:
        for tmp2 in tmp_chunk_list:
            if tmp2.upper() == tmp.upper():
                tmp_chunk_list.remove(tmp2)

# Remove Cities from main chunk list
for lst_city in lst_popular_cities:
    for tmp in continuous_chunk:
        if tmp.upper() in lst_city:
            if tmp.strip() in tmp_chunk_list:
                tmp_chunk_list.remove(tmp)

print(lst_popular_cities)
print(lst_states)
print(tmp_chunk_list)

'''
# COMMENT OUT FOR NOW TO AVOID ANY LIMITS
# GEOCODE
def do_geocode(address):
    try:
        return geolocator.geocode(address)
    except GeocoderTimedOut:
        return do_geocode(address)

geolocator = Nominatim(user_agent='geo_nlp')
all_locations = []
# States
for state in lst_states:
    state_usa = state + ', USA'
    location = ''
    try:
        location = geolocator.geocode(state_usa)
    except GeocoderTimedOut:
        location = do_geocode(state_usa)
    #print(location.raw)	
    lat_long = [location.latitude, location.longitude]
    all_locations.append(lat_long)
    #print(location.longitude, location.latitude, location.address)
# Cities
for lst_city in lst_popular_cities:
    city_name = lst_city[0]
    province_name = lst_city[1]
    country_name = lst_city[2]

    # Concatenate city, province, country name
    full_name = city_name + ', ' + province_name + ', ' + country_name

    location = ''
    try:
        location = geolocator.geocode(full_name)
    except GeocoderTimedOut:
        location = do_geocode(full_name)
    #print(location.raw)
    # Create multi-dimensional matrix for clustering
    lat_long = [location.latitude, location.longitude]
    all_locations.append(lat_long)
    #print(location.longitude, location.latitude, location.address)

'''

# GEOGRAPHIC SCOPE/ CLUSTERING
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
def most_common(List): 
    return(mode(List)) 

# TEST - Copy of the coordinates from the geocode
all_locations = [[37.1232245, -78.4927721], [39.7837304, -100.4458825], [31.8160381, -99.5120986], [37.1232245, -78.4927721], [37.1232245, -78.4927721], [37.5385087, -77.43428], [37.9357576, -122.3477486], [-41.3380953, 173.1872264], [39.8286897, -84.8898521], [37.7478572, -84.2946539], [-33.6009721, 150.7496405], [29.5821811, -95.7607832], [42.8091969, -82.7557554], [39.278622, -93.9768876], [41.4998322, -71.660263], [-20.7287264, 143.1414029], [41.8755616, -87.6244212], [32.7762719, -96.7968559], [44.9189206, -123.3158695], [33.9237141, -84.8407732], [29.7589382, -95.3676974], [37.2296566, -80.4136767]]

X = all_locations
X = StandardScaler().fit_transform(X)
#print(X)

# K-Means
print("CALCULATE K-MEANS")
print(X)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.labels_)
#print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)
# Get the mode for cluster with most frequent count
mode_cluster = most_common(kmeans.labels_)
# get the coordinates of item in cluster that most likely represent the area
like_cluster_pt = kmeans.cluster_centers_[mode_cluster]

# get the points from cluster with highest mode
print(mode_cluster)
cluster_most_labels = np.where(kmeans.labels_ == mode_cluster)
print(cluster_most_labels)

# Set all coordinates from the cluster label with most items
cluster_map = pd.DataFrame(all_locations) 
cluster_most = cluster_map[kmeans.labels_== mode_cluster] #np.extract(kmeans.labels_, X)
print(cluster_most)

 
'''
### DBSCAN ###
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
print(clustering.labels_)
print(clustering)

core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
labels = clustering.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(core_samples_mask)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


# Plot DBScan result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    print(class_member_mask)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

### END DBSCAN ###

'''

# GEOCODE LEFTOVERS
# Add Locations to LEFTOVERS and see if we can find a location or many locations


# LETS MAP THE DAMN THING


# map everything
# https://matplotlib.org/basemap/api/basemap_api.html
df = pd.DataFrame(all_locations) 
#print(df)
# BBox format is: BBox = ((df.longitude.min(),   df.longitude.max(), df.latitude.min(), df.latitude.max())
BBox = ((df[1].min(),   df[1].max(),    df[0].min(), df[0].max()))
#print(BBox)

# https://scitools.org.uk/cartopy/docs/v0.15/matplotlib/intro.html
# https://scitools.org.uk/cartopy/docs/v0.15/matplotlib/intro.html
ax = plt.axes(projection=ccrs.PlateCarree())
map_coast = ax.coastlines()
map_stock = ax.stock_img()

#https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db
ax.scatter(df[1], df[0], zorder=1, alpha= 0.2, c='r', s=50)
ax.set_title('Plotting All Toponyms')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
#ax.imshow(coast, zorder=0, extent = BBox, aspect= 'equal')

plt.show()

# Map only K-Means cluster with hight count
# Need to reset the BBox
print("MAP KMEANS")
print(cluster_most)
dfK = pd.DataFrame(cluster_most)
print(dfK)
BBox = ((dfK[1].min(),   dfK[1].max(),    dfK[0].min(), dfK[0].max()))
print(BBox)
ax = plt.axes(projection=ccrs.PlateCarree())
map_coast = ax.coastlines()
map_stock = ax.stock_img()

ax.scatter(dfK[1], dfK[0], zorder=1, alpha= 0.2, c='r', s=50)
ax.set_title('Plotting All Toponyms')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

plt.show()


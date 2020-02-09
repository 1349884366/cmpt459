#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def filtered_var(original,to_be_filter):
    for each in to_be_filter:
        original.remove(each)
        
    return original
    
def draw_graph(values):
    plt.style.use('ggplot')
    plt.hist(values, bins=100)
    plt.show()
    
    return

def get_bounds(variable):
	values = variable
	values.sort()
	length = len(values)
	inedx= int((length - 1)/4)
	q1 = (values[inedx] + values[inedx + 1])/2
	inedx = int((length - 1)*3/4)
	q3 = (values[inedx] + values[inedx + 1])/2
	iqr = q3 - q1
	lower_bound = q1 - 1.5*iqr            #100
	upper_bound = q3 + 1.5*iqr
	return lower_bound, upper_bound	

def get_outlier(variable):
    values = list(variable.values())
    lower_bound, upper_bound = get_bounds(values)
    outlier = []
    for i in values:
        if (i > upper_bound or i <lower_bound):
            outlier.append(i)
    return outlier

with open('train.json') as f:
    data = json.load(f)
    
    price = data['price']
latitude = data['latitude']
longitude = data['longitude']
time = data['created']

new_time = time
for keys in time:
    new_time[keys] = time[keys][11] + time[keys][12]

for keys in new_time:
    tmp = int(new_time[keys])
    new_time[keys] = tmp
    
draw_graph(list(price.values()))
draw_graph(list(latitude.values()))
draw_graph(list(longitude.values()))

x= list(time.values())
x.sort()
plt.xlim(1,24)
plt.style.use('ggplot')
plt.hist(x, bins=24)
plt.xticks(range(1,24))
plt.show()

top_5_hours = dict(Counter(new_time.values()).most_common(5))
other_hours = len(new_time) - sum(top_5_hours.values())
proportion={'other hours':other_hours}
for each in top_5_hours:
    proportion.update({each:top_5_hours[each]})

plt.pie(proportion.values(), labels=proportion.keys(),autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

df = pd.DataFrame(data)
outlier_price= get_outlier(price)
outlier_latitude = list(df[(df['latitude'].map(lambda d: d < 40))]['latitude'])
outlier_longitude = list(df[(df['longitude'].map(lambda d: d < -79.8 or d > -73.3))]['longitude'])

missing_value = df[(df['features'].map(lambda d: len(d)) == 0) | (df['display_address'].map(lambda d: d) == '') |
                    (df['description'].map(lambda d: d) == '') | (df['building_id'].map(lambda d: d) == '0') | (df['latitude'].map(lambda d: d) == 0) | (df['longitude'].map(lambda d: d) == 0)]
missing_feature_number = df[(df['features'].map(lambda d: len(d)) == 0)]['features'].count()
missing_display_address_number = df[(df['display_address'].map(lambda d: d) == '')]['display_address'].count()
missing_description_number = df[df['description'].map(lambda d: d) == '']['description'].count()
missing_building_id_number = df[df['building_id'].map(lambda d: d) == '0']['building_id'].count()
missing_latitude_number = df[(df['latitude'].map(lambda d: d) == 0)]['latitude'].count()
missing_longitude_number = df[(df['longitude'].map(lambda d: d) == 0)]['longitude'].count()
missing_street_address_number = df[(df['street_address'].map(lambda d: d) == '')]['street_address'].count()

filter_price = filtered_var(list(price.values()),outlier_price)
filter_latitude = filtered_var(list(latitude.values()),outlier_latitude)
filter_longitude = filtered_var(list(longitude.values()),outlier_longitude)

draw_graph(filter_price)
draw_graph(filter_latitude)
draw_graph(filter_longitude)

plt.boxplot(list(price.values()))
plt.show()
plt.boxplot(filter_price)
plt.show()
plt.boxplot(list(latitude.values()))
plt.show()
plt.boxplot(filter_latitude)
plt.show()
plt.boxplot(list(longitude.values()))
plt.show()
plt.boxplot(filter_longitude)
plt.show()
plt.scatter(range(1,len(price)+1),list(price.values()),alpha=0.5)
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

feature_content = []
for i in data['features'].values():
    feature_content += i
    
count_vect = CountVectorizer(ngram_range=(1,2), analyzer='word')
x_train_counts = count_vect.fit_transform(feature_content)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
total_count = 0
for each in count_vect.vocabulary_.values():
    total_count = total_count + each

feature_frequency_df = pd.DataFrame(x_train_tfidf.todense(), columns=count_vect.get_feature_names())

frequency_feature = {}
for keys in count_vect.vocabulary_.keys():
    frequency_feature.update({keys:float((count_vect.vocabulary_.get(keys)/total_count))})


# In[9]:





# In[ ]:


from PIL import Image
import requests
from io import BytesIO
from skimage import exposure
from skimage import feature

Hog_descriptor = data['photos']

for each1 in Hog_descriptor:
    descriptor_list = []
    for each2 in Hog_descriptor.get(each1):
        try:
            response = requests.get(each2)
            img = Image.open(BytesIO(response.content))
            width, height = img.size  
            left = 4
            top = height / 5
            right = 154
            bottom = 3 * height / 5
            newsize = (round(width/2), round(height/2))
            im1 = img.crop((left, top, right, bottom))
            im1 = im1.resize(newsize)
            H = feature.hog(im1, orientations=9, pixels_per_cell=(1, 1),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
            descriptor_list.append(H)
        except (OSError, NameError):
            print(each2)
    Hog_descriptor.update({each1:descriptor_list})


# In[25]:





# In[24]:





# In[6]:


outlier_price= get_outlier(price)
outlier_latitude = list(df[(df['latitude'].map(lambda d: d < 40))]['latitude'])
outlier_longitude = list(df[(df['longitude'].map(lambda d: d < -79.8 or d > -73.3))]['longitude'])


# In[21]:


outlier_price


# In[20]:


pd.DataFrame(data)


# In[ ]:





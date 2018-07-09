
# coding: utf-8

# # Project: Exploring the Movie Database (TMDB)
# ## Table of contents
# <ul>
# <li><a href="#intro"> Introduction </a></li>
# <li><a href="#wrangle"> Wrangling </a></li>
# <li><a href="#explore"> Expploratory data analysis </a></li>
# <li><a href="#conclude"> Conclusions </a></li>
# </ul>     

# <a id='intro'></a>
# ## Introduction

# I like action, animation, fantasy, and science fiction movies (not necessarily in that order!).
# For a personal project I would have liked the know what are the most popular movies in these categories. 
# Do other people like these categories as well?
# Do these movies make money?
# 
# But that would be a personal project. 
# I have refrained from these questions in the current project. 
# Instead I have followed the standard procedure of wrangling and exploring a given dataset. 
# Along the way I have asked questions based upon the features of the dataset. 
# 
# 1. What are the most popular movies of all time?
# 2. What are the features associated with these movies?
# 3. Who are the most popular directors?
# 4. How is the revenue distribution?
# 5. What are the top-grossing movies of all time?
# 6. Who are the directors of top-grossing movies?
# 7. Is popularity related to revenue?
# 8. Who are the most productive directors?
# 9. What's the yearly movie production rate?
# 10. Which genres have been popular over the years?
# 
# In the following, I first describe the data wrangling phase where I load the data into a dataframe, and then assess and clean. 
# Then I describe my explorations of the dataset. 
# I have explored *popularity, revenue, their interelation, directors, years, *and* genres.* 
# Along the way I also find the answers to the questions posed above. 

# ### Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# <a id='wrangle'></a>
# ## Wrangle

# ### Gather

# In[2]:


dataset = pd.read_csv('tmdb-movies.csv');
dataset.head()


# We see that there are 21 features associated with each entry. 
# The questions I am interested in primarily concerns the popularity and box-office performance of a movie. 
# In this respect, some of the features like 'id', 'imdb_id', and 'homepage' are irrelevant. 
# In the next phases of the wrangling procedure I shall consider modifying the dataframe to our need. 

# ### Assess

# Because a number of features could not be displayed abobe, let's first generate a list of the features in the dataset. 
# This information will be helpful in deciding which features to keep. 

# In[3]:


dataset.columns


# We can safely drop 'id', 'imdb_id', 'budget', 'revenue', and 'homepage' features: 'id' , 'imdb_id', and 'homepage' should have any relevance to the popularity and revenue of a movie, and raw 'budget' and 'revenue' are unnecessary as their adjusted values are also given in the final columns. 
# It may be argued that the tagline, keywords, and overview matters for the popularity and revenue of a movie. 
# However, this information may be redundant when the genres are specified or just too difficult to use. 
# In the following, I have therefore decided to drop these features as well. 
# Because the popularity score is already given, I shall not need 'vote_count' and 'vote_average'. 

# In[4]:


list_drop = ['id', 'imdb_id', 'budget', 'revenue', 'homepage', 'tagline', 'overview', 'vote_count', 'vote_average', 'keywords']
dataset.drop(list_drop, axis=1, inplace=True)
dataset.info()


# ### Clean 

# Time to clean data!
# We see that come entries for cast are missing. 
# Let's have a look at some of these movies. 

# #### Cast

# In[5]:


dataset[dataset.cast.isnull()].head()


# In[6]:


dataset[dataset.cast.isnull()].shape


# We see that adjusted budget and revenue seem to be missing as well. (They cannot be zero!) 
# Moreover, there are just 76 such items out of 10754, less than one percent. 
# I therefore drop these entries in the following. 

# In[7]:


dataset = dataset[pd.notnull(dataset['cast'])]
dataset.info()


# #### Director

# The next missing field is 'director'. 
# Let's have a look at the missing entries. 

# In[8]:


dataset[dataset.director.isnull()].head()


# In[9]:


dataset[dataset.director.isnull()].shape


# Budget and revenue are missing for these ones as well. 
# There are just 38 of them. 
# I shall remove them too. 

# In[10]:


dataset = dataset[pd.notnull(dataset['director'])]
dataset.info()


# #### Genre

# Some genre entries are missing. 

# In[11]:


dataset[dataset.genres.isnull()].head()


# In[12]:


dataset[dataset.genres.isnull()].shape


# And these too have missing budget and revenue and are few in numbers. 
# Drop them!

# In[13]:


dataset = dataset[pd.notnull(dataset['genres'])]
dataset.info()


# #### Production company

# Curiously a significant muber of movies have their production company missing. 
# Let's take a look. 

# In[14]:


dataset[dataset.production_companies.isnull()].head()


# In[15]:


dataset[dataset.production_companies.isnull()].shape


# While the budget and revenue information are missing for these ones as well, they are quite large in number. 
# I shall keep them for now. 
# I shall disregard them while exploring revenue. 

# <a id='explore'></a>
# ## Exploratory data analysis  

# ### 1. Popularity

# Let's start with popularity. 
# Because it is a float, we start with histogram analysis. 
# Our goal here is to see the distribution of popularity across movies. 

# In[16]:


dataset.popularity.hist()
plt.xlabel('Popularity')
plt.ylabel('Counts')
plt.title('Distribution of popularity');


# It seems like there are a outlies which have skewed the distribution. 
# To confirm this I use boxplot in the following with the "definite" outliers marked red. 

# In[17]:


dataset.popularity.plot(kind='box', vert=False, sym='r')
plt.xlabel('Popularity')
plt.title('Boxplot of popularity');


# This confirms my suspicion about outliers. 
# The outliers are the most popular movies in the database. 
# For the time-being I am interested in the distribution of *typical* popularity values. 
# To identify the typical points in shall use the five-point summary. 

# In[18]:


dataset.popularity.describe()


# #### Regular

# We shall now determine inner fences in boxplot to select the typical (regular) values. 

# In[19]:


# Quartiles and fences
q1_pop = dataset.popularity.describe()['25%']
q3_pop = dataset.popularity.describe()['75%']
iqr_pop = q3_pop - q1_pop
fence_low_pop = q1_pop - 1.5 * iqr_pop 
fence_high_pop = q3_pop + 1.5 * iqr_pop

# Regular and outlier samples
dataset_pop_reg = dataset.query('popularity > {} and popularity < {}'.format(fence_low_pop, fence_high_pop))
dataset_pop_out = dataset.query('popularity < {} or popularity > {}'.format(fence_low_pop, fence_high_pop))
dataset_pop_reg.shape


# We now plot the histogram of typical values of popularity. 

# In[20]:


dataset_pop_reg.popularity.plot.hist(bins=20)
plt.xlabel('Popularity')
plt.title('Distribution of "regular" values of popularity');


# We see that the typical value of popularity as around 0.3. 
# It is however heavily skewed to the right, meaning that more than half of the movies have popularity greater than this. 
# To get more exact figures we generate the statistics. 

# In[21]:


dataset_pop_reg.popularity.describe()


# #### Top 10!

# What are the top ten movies in terms of popularity? 
# This relates to our first few questions. 
# In the following we shall explore features associated with the most popular movies in the list. 

# In[22]:


dataset.sort_values(by='popularity', ascending=False).head(10)


# The list definitely makes sense (although I was somewhat surprised by the entry john Wick!). 
# This also answer to the first question: **What are the most popular movies of all time?** 
# In the following we explore the numerical features of these movies to answer our second question: **What are the features associated with these movies?**

# In[23]:


dataset.sort_values(by='popularity', ascending=False).head(10).describe()


# These are the numerical characteristics of the top ten popular movies. 
# Their runtime seems to a slightly high compared to other movies. 
# All of them are pretty recent with the exception of Star Wars (not surprisingly). 
# They all are high-budget movies the minimum being around 10 million USD and the median being around 140 million USD. 
# All them earned more than 70 million USD the highest one reaching almost 2.8 billion USD (Star Wars!). 

# #### Popular movie directors

# **Who are the most popular movie directors?** (Third question.) 
# There is no unique way to answer this, at least in terms of ranking. 
# I could just choose the directors of most popular movies and rank them according to the movies. 
# But this will be injustcie to directors who have many movies which are popular. 
# In the following I have considered 100 most popular movies and counted the directors. 
# Quentin Tanation and Christopher Nolan come on top, each delivering 5 among the 100 most popular movies. 

# In[24]:


dataset.sort_values(by='popularity', ascending=False).head(100).director.value_counts()[:10]


# ### 2. Revenue

# In our discussions on popularity, we saw that popular movies tend to make good money too. 
# In this part we shall explore the distribution of revenue across the movies in the database. 
# But first, lt us remember that there are many zero entries in the field. 
# Let us first see how many of them are there. 

# In[25]:


dataset.query('revenue_adj == 0').shape


# There's a lot of zeros! 
# Nevertheless, we still have more than 4000 movies with information on revenue. 
# We create a dataframe for the latter movies and explore. 

# #### Revenue proper

# In[26]:


dataset_rev = dataset.query('revenue_adj > 0')
dataset_rev.revenue_adj.hist()
plt.xlabel('Revenue (in 2010 USD)')
plt.ylabel('Counts')
plt.title('Distribution of revenue');


# We see that here too the distribution is affected by the presence of outliers. 
# To get a visual sense, we use boxplot as before. 

# In[27]:


dataset.revenue_adj.plot.box(vert=False, sym='r')
plt.xlabel('Revenue (in 2010 USD)')
plt.title('Boxplot of revenue');


# So there are many outliers. 
# To get a distribution for the regular entries, I use the parameters of boxplot to separate the outliers. 
# This is the same procedure as was the case with popularity. 

# In[28]:


# quartiles
q1_rev = dataset_rev.revenue_adj.describe()['25%']
q3_rev = dataset_rev.revenue_adj.describe()['75%']

# interquartile range
iqr_rev = q3_rev - q1_rev

# fences
fence_low_rev  = q1_rev - 1.5 * iqr_rev 
fence_high_rev = q3_rev + 1.5 * iqr_rev

# Regular and outlier samples
dataset_rev_reg = dataset_rev.query('revenue_adj > {} and revenue_adj < {}'.format(fence_low_rev, fence_high_rev))
dataset_rev_out = dataset_rev.query('revenue_adj < {} or revenue_adj > {}'.format(fence_low_rev, fence_high_rev))

# shape of the regular dataset
dataset_rev_reg.shape


# In[29]:


dataset_rev_reg = dataset_rev.query('revenue_adj > {} and revenue_adj < {}'.format(fence_low_rev, fence_high_rev))
dataset_rev_reg.revenue_adj.plot.hist(bins=20)
plt.xlabel('Revenue (in 2010 USD)')
plt.title('Distribution of available revenue');


# We see that unlike popularity histogram, there is no peak in the revenue distribution. 
# It's more skewed. 
# This also answers our fourth question: **How is the revenue distribution?** 
# From the above distribution I wondered if the same is true for the top-grossing movies (outliers). 

# In[30]:


dataset_rev_out = dataset_rev.query('revenue_adj < {} or revenue_adj > {}'.format(fence_low_rev, fence_high_rev))
dataset_rev_out.revenue_adj.plot.hist(bins=20)
plt.xlabel('Revenue (in 2010 USD)')
plt.title('Revenue distribution of top-grossing movies');


# Yes, the revenue distribution of the top-grossing movies is no different from the rest. 
# But **what are these top-grossing movies?** (Our fifth question.)
# I give the list of top ten movies in terms of revenue and their characteristics. 

# #### Top ten!

# In[31]:


dataset.sort_values(by='revenue_adj', ascending=False).head(10)


# In[32]:


dataset.sort_values(by='revenue_adj', ascending=False).head(50).describe()


# #### Highest grossing movie directors

# **Who directs these movies? ** (Our sixth question)
# Following the procedure for popularity we have the following list. 

# In[33]:


dataset.sort_values(by='revenue_adj', ascending=False).head(100).director.value_counts()[:10]


# Definitely not the same as before!

# ### 3. Popularity versus Revenue

# We saw that the most popular and most revenue generating movies need not overlap. 
# Here we want see if these is any correlation. 
# This is our seventh question:** Is popularity related to revenue?** 
# (It should be positive.)

# In[34]:


dataset_rev.plot(kind='scatter', x='popularity', y='revenue_adj');
plt.xlabel('Popularity')
plt.ylabel('Revenue (in 2010 USD)')
plt.title('Relation between popularity and revenue');


# As the scatter plot portrays, there is definitely a positive correlation. 
# The outliers have made the correlation a difficilt to see. 
# Also, there is a lot of spread. 
# In the following, we ask a related question: How strong is the correlation among the outliers? 
# If a movie is an outlier in popularity, what are the chances it will be an outlier in revenue, and vice versa?

# In[35]:


# sets of popular and revenue outliers
set_pop_out =  set(dataset_pop_out.original_title)
set_rev_out =  set(dataset_rev_out.original_title)
print('Popular outlier = {} and Revenue outlier = {}'
      .format(len(set_pop_out), len(set_rev_out)))


# In[36]:


# intersect of sets of popular and revenue outliers
print('Intersect: {}'.format(len(set_pop_out.intersection(set_rev_out))))


# A popular outlier has less than fift percent chance (331/930) of being a revenue outlier whereas a revenue outlier has more than fifty percent (331/470) chance of being a popular outlier. 

# ### 4. Directors

# Let's explore the data on directors. 
# (It's a categorical feature compared to the numerical features above.) 
# **Who are the most productive directors?** (Eighth question.) 

# In[37]:


dataset.director.value_counts()[:10]


# Not surprisingly, Woody Allen (45) comes at top followed by Clint Eastwood (34) and Steven Spielberg (20). 
# Spielberg is closely followed by Martin Scorses (28). 
# I wondered how many movies does a director generally make? 

# In[38]:


dataset.director.value_counts().describe()


# It seems that a director typically makes just one movie!

# ### 5. Year

#  ** What's the yearly movie production rate? ** (Question 9.) 
# This is best displayed graphically. 

# In[39]:


# select the year column
year = dataset.release_year

# value counts
year_counts = year.value_counts()

# reverse the value counts series to have the maximum on top
year_counts_rev = year_counts.reindex(year_counts.index[::-1])


# In[40]:


year_counts_rev.plot.barh(figsize=(8, 16))
plt.xlabel('Movies made')
plt.title('Yearly movie distribution');


# So the number movies per year is increasing steadily, though not always monotonically. For example, 2010 had less number of movies made than in year 2009 and 2008. Similarly, 2015 had less movies than 2013 and 2014.  

# ### 6. Genre

# Finally, we address the last question: 
# **Which genres are most popular from year to year?** 

# In[41]:


# earliest and latest years of record
earliest = int(dataset.release_year.describe()['min'])
latest = int(dataset.release_year.describe()['max'])

# For each year, collect the genres and find out
# the frequncy distribution
pop_gen = {}
for year in np.arange(earliest, latest+1):
    # collect the genre strings of a year as a series
    genres_year = dataset.query('release_year == {}'.format(year))['genres']
    genres = []
    # split and store genre strings in 'genres' list
    for genre in genres_year:
        genres.append(str(genre).split('|'))
    
    # make a flat list from 'genres' list 
    # so that we can determine frequencies
    flat_list = [item for sublist in genres for item in sublist]
    # make the flat list into a pandas series 
    # so that we can use the pd.value_counts() function
    genres = pd.Series(flat_list)
    pop_gen[str(year)] = genres.value_counts().idxmax()

# print the most popular genres per year
pop_gen


# Drama first! Comedy second. And none else! 
# This is al the more surprising from to the fact that thriller is the most common genre after drama, and not comedy, as we below. 

# In[42]:


pd.Series(flat_list).value_counts().plot(kind='pie', figsize=(10,10), label='Distribution of movie genres');


# <a id='conclude'></a>
# ## Conclusions

# In the above analysis I have explored popularity, revenue, their relation, directors, years, and genres in the movie database (TMDB). 
# The prime characteristic of the database is the presence of outlier. 
# In popularity, for example, we saw that a handful of movies reached a score as high as 10 whereas more than 3 out of 4 had a score below 1. 
# 
# Revenue had a lot of missing values. 
# Of the movies that did have the data, we saw that here too the presence of outliers was prominent. 
# In fact, the distribution seemed exponential, not even having a peak anywhere. 
# In a sense, there is no typical value of revenue. 
# 
# Next we considered the relation between popularity and revenue with mulple metrics. 
# They are definitely positively correlated on the average, although there is a lot of spread. 
# I compared most popular movies with highet-grossing movies. 
# A popular outlier has less than fift percent chance (331/930) of being a revenue outlier whereas a revenue outlier has more than fifty percent (331/470) chance of being a popular outlier.
# I also looked at the directors of these movies. 
# While there is a definite overlap there were some interesting mossions in each. 
# Quentin Tarantino came out in top as a popular movie make, but he was not among the top ten revenue generators. 
# Steven Spielberg came out in top as a revenue generator, but he was not among the top ten popular directors (in my interpretaion, of course!).
# 
# Just for fun, I also looked at the prolificity of directors. 
# Woody Allen (45) came in top followed by Clint Eastwood (34) and Steven Spielberg (20). Spielberg is closely followed by Martin Scorses (28). 
# 
# More an more movies are being made over the years. 
# The progression, however, is not strctly monotonic. 
# For example, 2010 had less number of movies made than in year 2009 and 2008. Similarly, 2015 had less movies than 2013 and 2014.  
# 
# Let me end the report with a curious observation about genres. 
# I looked at the most popular genre in a year. 
# Drama came first and comedy second. 
# And none else ever made in the list!
# This is all the more surprising becase comedy is not the second most frequent genre over all; it is thriller. 
# Yet, while comedy came out to be most popular genere many times, thriller never showed up. 
# Something to be investigated soon. 

# ### Limitations

# Any data-science investigation is limited by the quality of the dataset and the analysis. 
# The current investigation is no exception. 
# The dataset for the current investigation has entries for years 1960 to 2015. 
# We could not make any concrete comments about movies made before or afterwards. 
# 
# The revenue was not list for almost half of the movies. 
# So the statistics mentioned in relation to revenue, while representative, may not be robust. 
# This is especially true for the outliers for which we may have missed some top-grossing movies. 
# 
# While the popularity metric made sense in terms of the movies that we know to be popular, its definition was missing in the description of the dataset. 
# While it must be a reasonable function of the various features of a movie, we cannot be sure that it did not favor some types of movies over others. 
# 
# On the analysis side, I was not able to explore all the features involved in the dataset and their dependency. 
# Given time, it would have been interesting to explore, for example, the influence of cast and directors on popularity, the popularity of different genres, or if a particular genre is getting more and more popular, among others.  

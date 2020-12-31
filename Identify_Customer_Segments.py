#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing as p
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')


# In[3]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
azdias.head(10)


# In[4]:


feat_info.head(85)


# In[5]:


azdias.info()


# In[6]:


azdias.describe()


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[7]:


# Identify missing or unknown data values and convert them to NaNs.
# 1. Convert the strings of a list to actual lists

def string_to_list(x):
    list = []
    x = x.strip("][").split(",")
    for char in x:
        try:
            list.append(int(char))
        except:
            list.append(char)
    return list

    
practice_list = string_to_list("[-1,0]")
practice_list


# In[8]:


# 2. Applying the function
feat_info["NaN_Markers"] = feat_info["missing_or_unknown"].apply(string_to_list)
feat_info.head()


# In[164]:


na_azdias = azdias[:]


# In[165]:


# 3. Insert NA values into our DataFrame
attribute_index = feat_info.set_index('attribute')
na_azdias = azdias[:]

for column in na_azdias.columns:
    na_azdias[column].replace(attribute_index.loc[column].loc['NaN_Markers'],np.NaN,inplace=True)


# In[10]:


# Check
na_azdias.isna().sum().sum()


# In[11]:


na_azdias.head()


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[12]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

missing_data = na_azdias.isna().sum()
missing_data_rank = missing_data.reset_index()
missing_data_rank.sort_values(by=0, ascending=False,inplace=True)
missing_data_rank.head(50)


# In[13]:


# Investigate patterns in the amount of missing data in each column (2).
highest_NaN_cols = dict()
lower_NaN_cols = dict()
for column in na_azdias:
    sum_by_cols = na_azdias[column].isna().sum()
    if sum_by_cols > 250000:
        highest_NaN_cols[column] = sum_by_cols
    else:
        lower_NaN_cols[column] = sum_by_cols
highest_NaN_cols


# In[14]:


azdias.shape[0]


# In[15]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
hurdle_25 = azdias.shape[0]*0.25
hurdle_10 = azdias.shape[0]*0.10
hurdle_01 = azdias.shape[0]*0.01


# In[16]:


outliers = missing_data[missing_data > hurdle_25]
outliers


# In[17]:


moderate_NaN = missing_data[missing_data > hurdle_10]
print(moderate_NaN)


# In[18]:


low_NaN = missing_data[missing_data > 0]
clean_cols = missing_data[missing_data == 0]
len(clean_cols)
print(f"Columns over {np.round(hurdle_25,0)} missing records: {len(outliers)}")
print(f"Columns with between {np.round(hurdle_10,0)} and {np.round(hurdle_25,0)} missing records: {len(moderate_NaN)-len(outliers)}")
print(f"Columns with up to {np.round(hurdle_10,0)} missing records: {len(low_NaN)-len(moderate_NaN)}")
print(f"Cols with no missing data: {len(clean_cols)}")


# In[19]:


missing_data[missing_data == 116515]


# In[20]:


missing_data[missing_data == 133324]


# In[21]:


missing_data[missing_data == 93148]


# In[22]:


missing_data[missing_data == 158064]


# In[123]:


clean_azdias = na_azdias.drop(labels=outliers.index, axis = 1)
print(clean_azdias.shape)


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# *(Double click this cell and replace this text with your own text, reporting your observations regarding the amount of missing data in each column. Are there any patterns in missing values? Which columns were removed from the dataset?)*
# 
# A brief missing values analysis shows that the dataset has 32 fields with no missing data, 14 with up to 89 thousand missing values (10% of records), 35 with between 89 k  and 222 k missing records (between 10% and 25%), and 4 columns with more than 25% of missing values, or over 222,805 records.
# 
# Also worth noting that there are several clusters of fields with the same numbers of missing values, indicating relationships underlying those fields, including what may represent regional differences in data quality or simply that data is collected in different regions and the corresponding fields are empty for records from other regions. Some are also rlated to different attributes regarding the same variable (3. Building-level features). 
# 
# Some of the repeated missing values frequencies are displayed below:
# 
# Regional features
# 
#     PLZ8_ANTG1     116515
#     PLZ8_ANTG2     116515
#     PLZ8_ANTG3     116515
#     PLZ8_ANTG4     116515
#     PLZ8_BAUMAX    116515
#     PLZ8_HHZ       116515
#     PLZ8_GBZ       116515
# 
#     KBA05_ANTG1    133324
#     KBA05_ANTG2    133324
#     KBA05_ANTG3    133324
#     KBA05_ANTG4    133324
#     KBA05_GBZ      133324
#     MOBI_REGIO     133324
# 
# Building-level features
# 
#     GEBAEUDETYP         93148
#     MIN_GEBAEUDEJAHR    93148
#     OST_WEST_KZ         93148
#     WOHNLAGE            93148
# 
# Others:
# 
#     KKK         158064
#     REGIOTYP    158064
#     
# The top 5 missing values variables, with over 222.8 thousand (or 25% of total records) have been removed from the clean dataset, and are displayed next:
# 
#     TITEL_KZ        889061    (Academic title)
#     AGER_TYP        685843    (Best-ager typology)
#     GEBURTSJAHR     392318    (Year of birth)
#     ALTER_HH        310267    (birthday of head of household)
#     KK_KUNDENTYP    584612    (Consumer pattern over past 12 months)
#     KBA05_BAUMAX    476524    (regional variable)
#     
# However, regional patterns in data completeness may indicate that the "KBA05_BAUMAX" variable should not be removed if one were to perform an analysis on a region by region basis.

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[24]:


# How much data is missing in each row of the dataset?
missing_by_row = clean_azdias.isna().sum(axis=1)
plt.hist(missing_by_row, bins=50);


# In[116]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
cutoff = 20
azdias_zero_md = clean_azdias[clean_azdias.isna().sum(axis=1) == 0]
azdias_more_than_0 = clean_azdias[clean_azdias.isna().sum(axis=1) > 0]
azdias_1_to_20 = clean_azdias[(clean_azdias.isna().sum(axis=1) >= 1) & (clean_azdias.isna().sum(axis=1) <= cutoff)]
azdias_less_than_20 = clean_azdias[clean_azdias.isna().sum(axis=1) <= cutoff]
azdias_more_than_20 = clean_azdias[clean_azdias.isna().sum(axis=1) > cutoff]

azdias_zero_md.describe()


# In[122]:


print('Rows with no missing data: ',azdias_zero_md.shape[0])
print('Rows with missing data: ',azdias_more_than_0.shape[0])
print('% of rows with missing data: ', np.round( 100 * (azdias_more_than_0.shape[0] / azdias.shape[0]), 2))


# In[26]:


compare_columns = ["MOBI_REGIO", "PLZ8_BAUMAX", "GEBAEUDETYP", "KKK", "REGIOTYP", "WOHNLAGE", "ANZ_PERSONEN", "LP_FAMILIE_GROB"]


# In[27]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.

figure, axs = plt.subplots(nrows=len(compare_columns), ncols=2, figsize = (15,20))
figure.subplots_adjust(hspace = 1, wspace=.3)
for row in range(len(compare_columns)):
    sns.countplot(azdias_1_to_20[compare_columns[row]], ax=axs[row][0])
    axs[row][0].set_title('1 to 20 Missing Values')
    sns.countplot(azdias_more_than_20[compare_columns[row]], ax=axs[row][1])
    axs[row][1].set_title('>20 missing values')


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# *(Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)*
# 
# The 8 fields used for the sample are among the surviving fields with highest numbers of missing values.
# 
# The low missing values subset has been created excluding rows with no missing values.
# 
# Despite that most of them display similar distributions between subsets, the following fields show sufficient distributional differences that allow to refrain from eliminating them in order to avoid information loss: MOBI_REGIO, WOHNLAGE.
# 
# On the other hand, despite that only 30% of rows contain any missing values, in order to avoid excessive information loss, any rows with remaining missing values will be eliminated only **after** the removal of all columns deemed unfit for analysis.

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[28]:


# How many features are there of each data type?
feat_info.type.value_counts()


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[79]:


data_type_dictionary = {}
data_types = ['categorical','mixed','numeric','ordinal']

for i in data_types:
    data_type_dictionary[i] = feat_info[ feat_info['type'] == i ]['attribute']


# In[80]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

categorical_fds = feat_info[feat_info['type']=='categorical']
categorical_fds


# In[81]:


# Data type dictionary mapped to corresponding fields in DB
mapping_categoricals_to_df = azdias[data_type_dictionary['categorical']]


# In[82]:


# Display categories present in multi level categorical fields
multi_level_categorical = []
binary_categorical = []

for column in mapping_categoricals_to_df.columns:
        original = mapping_categoricals_to_df[column]
        unique_values = original[original.notna()].unique()
        count_unique = len( unique_values )
        
        if count_unique >2:
            multi_level_categorical.append(column)
        elif count_unique == 2:
            binary_categorical.append(column)
        
        print(column, unique_values, count_unique)


# In[83]:


# Display fields by type

print('Binary fields:')
for column in binary_categorical:
    print(column) 

print('\n', 'Multi level categorical fields:')
for column in multi_level_categorical:
    print(column)

print('\n', 'Mixed fields:')
for column in data_type_dictionary['mixed']:
    print(column)


# #### Features for recoding
# 
# * Non numeric binary: OST_WEST_KZ
# * All multilevel categoricals

# In[84]:


# Re-encode categorical variable(s) to be kept in the analysis.

# 1. Select variables to be re-encoded

reencode_variables = []
reencode_variables.append('OST_WEST_KZ')  # Non-numeric binary manually included

# for column in data_type_dictionary['mixed']:  # Removed all mixed variables from encoding
#    reencode_variables.append(column)
    
for column in multi_level_categorical:
    reencode_variables.append(column)
    
for i in  reencode_variables:
    if   i == 'AGER_TYP' or i =='TITEL_KZ' or  i =='ALTER_HH' or  i =='KK_KUNDENTYP' or  i =='KBA05_BAUMAX' or i =='LP_LEBENSPHASE_FEIN' or i =='LP_STATUS_FEIN' or i =='LP_LEBENSPHASE_GROB' or i =='PRAEGENDE_JUGENDJAHRE' or i =='WOHNLAGE' or i =='CAMEO_INTL_2015' or i =='PLZ8_BAUMAX' or i == 'CAMEO_DEU_2015' or i == 'NATIONALITAET_KZ': # Remove variables previously dropped
        reencode_variables.remove(i)
        
reencode_variables


# In[124]:


print(clean_azdias.shape)


# In[125]:


# 2. Dropping variables from data
# Select variables to drop
features_to_drop_list = ['LP_LEBENSPHASE_FEIN', 'LP_STATUS_FEIN', 'CAMEO_DEU_2015', 'NATIONALITAET_KZ']
features_to_drop = pd.DataFrame(features_to_drop_list).iloc[0:len(features_to_drop_list),-1]
print(features_to_drop)


# In[126]:


# Drop them
clean_azdias_df = clean_azdias.drop(labels=features_to_drop, axis = 1)
print(clean_azdias_df.shape)


# In[127]:



# Remove mixed variables that will not be used (see later in point 1.2.2)

mixed_remove = ['LP_LEBENSPHASE_GROB', 'PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015', 'PLZ8_BAUMAX']
clean_azdias_df = clean_azdias_df.drop(labels=mixed_remove, axis = 1)

print(clean_azdias_df.shape)


# In[128]:


# 3. Re-encoding (one hot)

clean_azdias_df_dum = pd.get_dummies(clean_azdias_df, columns=reencode_variables)


# In[129]:


print(clean_azdias_df_dum.shape)


# In[130]:


clean_azdias_df_dum.head()


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# *(Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)*
# 
# 1. Breakdown by variable type:
# 
#         ordinal        49
#         categorical    21 : of which 'TITEL_KZ' and 'KK_KUNDENTYP' had been dropped because of many missing values
#         numeric         7
#         mixed           7
#         interval        1
# 
# 2. Breakdown of categorical variables:
#     a. Binary (4): Out of the byinary fields, 'OST_WEST_KZ' is non-numeric (values: ['W' 'O']), and required re encoding.
#     
#         ANREDE_KZ
#         GREEN_AVANTGARDE
#         SOHO_KZ
#         OST_WEST_KZ (*)
#     
#     b. Multilevel categoricals (17): of which *'TITEL_KZ'* and *'KK_KUNDENTYP'* had already been dropped because of too many missing values.
#         
#         AGER_TYP
#         CJT_GESAMTTYP
#         FINANZTYP
#         GFK_URLAUBERTYP
#         LP_FAMILIE_FEIN
#         LP_FAMILIE_GROB
#         LP_STATUS_FEIN
#         LP_STATUS_GROB
#         NATIONALITAET_KZ
#         SHOPPER_TYP
#         TITEL_KZ (*)
#         VERS_TYP
#         ZABEOTYP
#         KK_KUNDENTYP (*)
#         GEBAEUDETYP
#         CAMEO_DEUG_2015
#         CAMEO_DEU_2015
# 
#     c. Mixed variables (7): of which *'KBA05_BAUMAX'* had already been dropped because of too many missing values.
#         
#         LP_LEBENSPHASE_FEIN
#         LP_LEBENSPHASE_GROB
#         PRAEGENDE_JUGENDJAHRE
#         WOHNLAGE
#         CAMEO_INTL_2015
#         KBA05_BAUMAX (*)
#         PLZ8_BAUMAX
#         
#      d. Additional features to exclude: 
#      
#       i. I selected most of the attributes ending in 'FEIN' to be excluded, because those fields already are present in a compressed manner in other fields ending in 'GROB', and their remotion will not imply a significant loss of information, but will represent a large reduction in dataset size after reencoding. Under this concept, the following variables were removed:
# 
#         LP_LEBENSPHASE_FEIN
#         LP_STATUS_FEIN
#         
#         
#      ii. Variables that are redundant and removed before dummy generation: 
#         
#         CAMEO_DEU_2015
#      
#          

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# #### Work notes:
# 
# Checking into the field descriptions in the Data Dictionary:
# 
# ###### 1.18. PRAEGENDE_JUGENDJAHRE:
# 
# Dominating movement of person's youth (avantgarde vs. mainstream; east vs. west)
# - -1: unknown (*)
# -  0: unknown (*)
# -  1: 40s - war years (Mainstream, E+W)
# -  2: 40s - reconstruction years (Avantgarde, E+W)
# -  3: 50s - economic miracle (Mainstream, E+W)
# -  4: 50s - milk bar / Individualisation (Avantgarde, E+W)
# -  5: 60s - economic miracle (Mainstream, E+W)
# -  6: 60s - generation 68 / student protestors (Avantgarde, W)
# -  7: 60s - opponents to the building of the Wall (Avantgarde, E)
# -  8: 70s - family orientation (Mainstream, E+W)
# -  9: 70s - peace movement (Avantgarde, E+W)
# - 10: 80s - Generation Golf (Mainstream, W)
# - 11: 80s - ecological awareness (Avantgarde, W)
# - 12: 80s - FDJ / communist party youth organisation (Mainstream, E)
# - 13: 80s - Swords into ploughshares (Avantgarde, E)
# - 14: 90s - digital media kids (Mainstream, E+W)
# - 15: 90s - ecological awareness (Avantgarde, E+W)
# 
# 
# ###### 4.3. CAMEO_INTL_2015
# German CAMEO: Wealth / Life Stage Typology, mapped to international code
# - -1: unknown (*)
# - 11: Wealthy Households - Pre-Family Couples & Singles
# - 12: Wealthy Households - Young Couples With Children
# - 13: Wealthy Households - Families With School Age Children
# - 14: Wealthy Households - Older Families &  Mature Couples
# - 15: Wealthy Households - Elders In Retirement
# - 21: Prosperous Households - Pre-Family Couples & Singles
# - 22: Prosperous Households - Young Couples With Children
# - 23: Prosperous Households - Families With School Age Children
# - 24: Prosperous Households - Older Families & Mature Couples
# - 25: Prosperous Households - Elders In Retirement
# - 31: Comfortable Households - Pre-Family Couples & Singles
# - 32: Comfortable Households - Young Couples With Children
# - 33: Comfortable Households - Families With School Age Children
# - 34: Comfortable Households - Older Families & Mature Couples
# - 35: Comfortable Households - Elders In Retirement
# - 41: Less Affluent Households - Pre-Family Couples & Singles
# - 42: Less Affluent Households - Young Couples With Children
# - 43: Less Affluent Households - Families With School Age Children
# - 44: Less Affluent Households - Older Families & Mature Couples
# - 45: Less Affluent Households - Elders In Retirement
# - 51: Poorer Households - Pre-Family Couples & Singles
# - 52: Poorer Households - Young Couples With Children
# - 53: Poorer Households - Families With School Age Children
# - 54: Poorer Households - Older Families & Mature Couples
# - 55: Poorer Households - Elders In Retirement
# - XX: unknown (*)
# 
# (*) Removed
# 

# In[131]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
# 1. Investigate

print(clean_azdias.PRAEGENDE_JUGENDJAHRE.unique())

# 2. Create new mappings (dicts):
# Mappings for 'Decade' are obvious:
praegende_jugendjahre_decade = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60, 8: 70, 9:70, 10:80, 11:80, 12: 80, 13:80, 14:90, 15:90}
praegende_jugendjahre_movement = {1:0, 2:1, 3:0, 4:1, 5:0, 6:1, 7:1 ,8:0 ,9:1, 10:0, 11:1, 12:0 ,13:1 ,14:0, 15:1}

# Mappings 'Movement':
#          Mainstream: 0 
#          Avantgarde: 1


# In[132]:


# 3. Create new columns: Note that the original variable had been removed before onehot encoding

clean_azdias_df_dum["P_J_DECADE_synth"] = clean_azdias["PRAEGENDE_JUGENDJAHRE"].map(praegende_jugendjahre_decade, na_action = "ignore")
clean_azdias_df_dum["P_J_MOVEMENT_synth"] = clean_azdias["PRAEGENDE_JUGENDJAHRE"].map(praegende_jugendjahre_movement, na_action = "ignore")
print(clean_azdias_df_dum.P_J_DECADE_synth.unique())
print(clean_azdias_df_dum.P_J_MOVEMENT_synth.unique())


# In[133]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
# 1. Investigate

print(clean_azdias.CAMEO_INTL_2015.unique())


# In[134]:


#2. Create new mappings:

CAMEO_Wealth = {'11': 1, '12': 1, '13': 1, '14': 1, '15': 1, '21': 2, '22': 2, '23': 2, '24': 2, '25': 2, '31': 3, '32': 3, '33': 3, '34': 3, '35': 3, '41': 4, '42': 4, '43': 4, '44': 4, '45': 4, '51': 5, '52': 5, '53': 5, '54': 5, '55': 5}
CAMEO_Stage = {'11': 1, '12': 2, '13': 3, '14': 4, '15': 5, '21': 1, '22': 2, '23': 3, '24': 4, '25': 5, '31': 1, '32': 2, '33': 3, '34': 4, '35': 5, '41': 1, '42': 2, '43': 3, '44': 4, '45': 5, '51': 1, '52': 2, '53': 3, '54': 4, '55': 5}


# In[135]:


# 3. Create new columns: Note that the original variable had been removed before onehot encoding

clean_azdias_df_dum["CAMEO_Wealth_synth"] = clean_azdias["CAMEO_INTL_2015"].map(CAMEO_Wealth, na_action = "ignore")
clean_azdias_df_dum["CAMEO_Stage_synth"] = clean_azdias["CAMEO_INTL_2015"].map(CAMEO_Stage, na_action = "ignore")
print(clean_azdias_df_dum.CAMEO_Wealth_synth.unique())
print(clean_azdias_df_dum.CAMEO_Stage_synth.unique())


# In[47]:


# Remove old columns: 'CAMEO_INTL_2015' and 'PRAEGENDE_JUGENDJAHRE' were removed before creating dummy variables


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# *(Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)*
# 
# Firstly, I went through the Data Dictionary to inspect the first two mixed variables, to identify their underlying segmentation. This lead to breaking down each original variable into two new synthetic variables which consider the most relevant dimensions in the old ones:
# 
#  * PRAEGENDE_JUGENDJAHRE into P_J_DECADE_synth and P_J_MOVEMENT_synth, by mapping the generational and movement subfeatures into new variables. I performed a manual mapping array due to time constraints, to be mapped into the dataset with the .map method.
#  
#  * CAMEO_INTL_2015 into CAMEO_Wealth_synth and CAMEO_Stage_synth, by mapping the wealth and life stage subfeatures into new variables. I performed a manual mapping array due to time constraints, to be mapped into the dataset with the .map method.
#   
# In addition to the two original mixed type variables that were re-engineered, I decided to remove the remaining ones from the dataset (which I can not process due to time constraints). The implementation consisted on going back to the stage prior to the creation of the dummy variables, and deleting the mixed variables before they were broken into multiple dummy variables, which were going to become much more complicated to deal with.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[136]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)

for column in clean_azdias_df_dum.columns:
    print(column, clean_azdias_df_dum[column].unique())


# In[137]:


# Current table dimensions

print('Shape: {}, No. of data points: {}' .format(clean_azdias_df_dum.shape, clean_azdias_df_dum.size))


# In[141]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.
# 1. Select columns to drop on a qualitative basis:
#    - ALTERSKATEGORIE_GROB (Est. age based on given name analysis): Can be very imprecise and is edundant with ALTER_HH
#    - CAMEO_DEU_2015 (Detailed wealth & lifestile): Compressed in CAMEO_DEUG_2015 (Rough wealth & lifestile). Is removed from pre dummy generation
#    - FINANZ_MINIMALIST, FINANZ_SPARER, FINANZ_VORSORGER, FINANZ_ANLEGER, FINANZ_UNAUFFAELLIGER, FINANZ_HAUSBAUER (qualification per financial typology): Redundant with the compressed variable FINANZTYP
#    - GREEN_AVANTGARDE: Redundant with P_J_MOVEMENT_synth

quali_drop_list = ['ALTERSKATEGORIE_GROB', 'FINANZ_MINIMALIST', 'FINANZ_SPARER', 'FINANZ_VORSORGER', 'FINANZ_ANLEGER', 'FINANZ_UNAUFFAELLIGER', 'FINANZ_HAUSBAUER', 'GREEN_AVANTGARDE']
clean_azdias_flds = clean_azdias_df_dum.drop(labels=quali_drop_list, axis = 1)

print('Rows: {}, Columns: {}, No. of data points: {}' .format(clean_azdias_flds.shape[0], clean_azdias_flds.shape[1], clean_azdias_flds.size))


# In[167]:


# I will finally drop all rows with missing values, after all necessary columns have been dropped:
    
clean_azdias_more_than_0 = clean_azdias_flds[clean_azdias_flds.isna().sum(axis=1) > 0]
clean_azdias_zero_md = clean_azdias_flds[clean_azdias_flds.isna().sum(axis=1) == 0]
    
print('Before:')
print('Rows with no missing data: ', clean_azdias_zero_md.shape[0])
print('Rows with missing data: ', clean_azdias_more_than_0.shape[0])
print('% of rows with missing data: ', np.round( 100 * (clean_azdias_more_than_0.shape[0] / azdias.shape[0]), 2))

clean_azdias_flds['number_missing'] = clean_azdias_flds.isna().sum(axis=1)

clean_azdias_flds_rows = clean_azdias_flds[clean_azdias_flds['number_missing'] == 0]
clean_azdias_flds_rows.drop('number_missing', inplace=True, axis=1)


clean_azdias_with_md = clean_azdias_flds_rows[clean_azdias_flds_rows.isna().sum(axis=1) > 0]
clean_azdias_no_md = clean_azdias_flds_rows[clean_azdias_flds_rows.isna().sum(axis=1) == 0]

print('After:')
print('Rows with no missing data: ', clean_azdias_no_md.shape[0])
print('Rows with missing data: ', clean_azdias_with_md.shape[0])
print('% of rows with missing data: ', np.round( 100 * (clean_azdias_with_md.shape[0] / azdias.shape[0]), 2))


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[168]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    
    # 0.1.1 Load feat_info:
    
    feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')
    
    # 0.1.2 Initial data state analysis:
    
    print('>> 0. Initial data state analysis:')
    
    print('Rows: {}, Columns: {}, No. of data points: {}' .format(df.shape[0], df.shape[1], df.size))    
        
        
    # 1. Preprocessing:
    
    # 1.1 Assess missing data
    # 1.1.1 convert missing value codes into NaNs
    
    feat_info["NaN_Markers"] = feat_info["missing_or_unknown"].apply(string_to_list)
    
    attribute_index = feat_info.set_index('attribute')
    df_na = df[:]

    for column in df_na.columns:
        df_na[column].replace(attribute_index.loc[column].loc['NaN_Markers'],np.NaN,inplace=True)
    
    initial_md_count = df_na.isna().sum().sum()
    initial_md_to_total_ratio = initial_md_count / df_na.size
    
    print('>> 1.1. Initial missing data: Missing values: ')
    print('{}, No. of data points: {}, Ratio to total data points %: {}' .format(initial_md_count, df_na.size, np.round(100*initial_md_to_total_ratio,2)))
   

    # 1.1.2 remove selected columns and rows
    # 1.1.2.a Remove outlier columns
    hurdle_25 = df.shape[0]*0.25
    hurdle_10 = df.shape[0]*0.10
    hurdle_01 = df.shape[0]*0.01
    
    missing_data = df.isna().sum()
    outliers = missing_data[missing_data > hurdle_25]
    moderate_NaN = missing_data[missing_data > hurdle_10]
    low_NaN = missing_data[missing_data > 0]
    clean_cols = missing_data[missing_data == 0]
    len(clean_cols)
    
    print('>> 1.1.2 Missing data by columns:')
    print(f"Columns over {np.round(hurdle_25,0)} missing records: {len(outliers)}")
    print(f"Columns with between {np.round(hurdle_10,0)} and {np.round(hurdle_25,0)} missing records: {len(moderate_NaN)-len(outliers)}")
    print(f"Columns with up to {np.round(hurdle_10,0)} missing records: {len(low_NaN)-len(moderate_NaN)}")
    print(f"Cols with no missing data: {len(clean_cols)}")
    
    clean_df1 = df_na.drop(labels=outliers.index, axis = 1)
    
    # 1.1.2.b Remove redundant and low info variables:
    features_to_drop_list = ['LP_LEBENSPHASE_FEIN', 'LP_STATUS_FEIN', 'CAMEO_DEU_2015', 'NATIONALITAET_KZ']
    features_to_drop = pd.DataFrame(features_to_drop_list).iloc[0:len(features_to_drop_list),-1]
    
    clean_df2 = clean_df1.drop(labels=features_to_drop, axis = 1)
    
    # 1.1.2.c Remove remaining mixed type variables:
    
    mixed_remove = ['LP_LEBENSPHASE_GROB', 'PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015', 'PLZ8_BAUMAX']
    clean_df3 = clean_df2.drop(labels=mixed_remove, axis = 1)
    
    # 1.1.2.d Remove variables on a qualitative basis:
    
    quali_drop_list = ['ALTERSKATEGORIE_GROB', 'FINANZ_MINIMALIST', 'FINANZ_SPARER', 'FINANZ_VORSORGER', 'FINANZ_ANLEGER', 'FINANZ_UNAUFFAELLIGER', 'FINANZ_HAUSBAUER', 'GREEN_AVANTGARDE']
    clean_df4 = clean_df3.drop(labels=quali_drop_list, axis = 1)
    
    print('>> 1.1.2 Preprocessed data:')
    print('Rows: {}, Columns: {}, No. of data points: {}' .format(clean_df3.shape[0], clean_df3.shape[1], clean_df3.size))
    
    preprocessed_md_count = clean_df4.isna().sum().sum()
    preprocessed_md_to_total_ratio = preprocessed_md_count / clean_df4.size
    print('Preprocessed missing data: Missing values: {}, No. of data points: {}, Ratio to total data points %: {}' .format(preprocessed_md_count, clean_df4.size, np.round(100*preprocessed_md_to_total_ratio,2)))   
    
    # 1.1.3 Remove rows with missing values

    clean_df4['number_missing'] = clean_df4.isna().sum(axis=1)
    clean_df_flds_rows = clean_df4[clean_df4['number_missing'] == 0]
    clean_df_flds_rows.drop('number_missing', inplace=True, axis=1)


    clean_df_with_md = clean_df_flds_rows[clean_df_flds_rows.isna().sum(axis=1) > 0]
    clean_df_no_md = clean_df_flds_rows[clean_df_flds_rows.isna().sum(axis=1) == 0]

    print('>> After removing all remaining rows with missing data:')
    print('Rows with no missing data: ',clean_df_no_md.shape[0])
    print('Rows with missing data: ',clean_df_with_md.shape[0])
    print('% of rows with missing data: ', np.round( 100 * (clean_df_with_md.shape[0] / clean_df_flds_rows.shape[0]), 2))
    
        
    # 1.2 Select, re-encode, and engineer column values.
    
    # 1.2.1 Select and re-encode features
    # 1.2.1.a. Create data type dictionary
    
    data_type_dictionary = {}
    data_types = ['categorical','mixed','numeric','ordinal']

    for i in data_types:
        data_type_dictionary[i] = feat_info[ feat_info['type'] == i ]['attribute']

    # 1.2.1.b. Select encoding targets  
    
        # i. Categorical multi-level variables:
    
    categorical_fds = feat_info[feat_info['type']=='categorical']
    mapping_categoricals_to_df = df[data_type_dictionary['categorical']]
    
    multi_level_categorical = []
    binary_categorical = []

    for column in mapping_categoricals_to_df.columns:
            original = mapping_categoricals_to_df[column]
            unique_values = original[original.notna()].unique()
            count_unique = len( unique_values )
        
            if count_unique >2:
                multi_level_categorical.append(column)
            elif count_unique == 2:
                binary_categorical.append(column)

        # ii. Categorical variable(s) to be reencoded
                
    reencode_variables = []
    reencode_variables.append('OST_WEST_KZ')    # Non-numeric binary variable manually included

    for column in multi_level_categorical:
        reencode_variables.append(column)
    
    for i in  reencode_variables:
        if   i == 'AGER_TYP' or i =='TITEL_KZ' or  i =='ALTER_HH' or  i =='KK_KUNDENTYP' or  i =='KBA05_BAUMAX' or i =='LP_LEBENSPHASE_FEIN' or i =='LP_STATUS_FEIN' or i =='LP_LEBENSPHASE_GROB' or i =='PRAEGENDE_JUGENDJAHRE' or i =='WOHNLAGE' or i =='CAMEO_INTL_2015' or i =='PLZ8_BAUMAX' or i == 'CAMEO_DEU_2015' or i == 'NATIONALITAET_KZ': # Remove variables previously dropped
            reencode_variables.remove(i)
                
    # 1.2.1.c. Reencode selected variables:
    
    clean_df_dum_zero = pd.get_dummies(clean_df_flds_rows, columns=reencode_variables)
    
    print('>> 1.2.1 Preprocessed data with dummies:')
    print('Rows: {}, Columns: {}, No. of data points: {}' .format(clean_df_dum_zero.shape[0], clean_df_dum_zero.shape[1], clean_df_dum_zero.size))
    
    preprocessed_md_count_wd = clean_df_dum_zero.isna().sum().sum()
    preprocessed_md_to_total_ratio_wd = preprocessed_md_count_wd / clean_df_dum_zero.size
    print('Preprocessed missing data: Missing values: {}, No. of data points: {}, Ratio to total data points %: {}' .format(preprocessed_md_count_wd, clean_df_dum_zero.size, np.round(100*preprocessed_md_to_total_ratio_wd,2)))    
    
    
    # 1.2.2 Engineer mixed-type features
    
    # 1.2.2.a Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
    
        # i. Define mappings:
    
    praegende_jugendjahre_decade = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60, 8: 70, 9:70, 10:80, 11:80, 12: 80, 13:80, 14:90, 15:90}
    praegende_jugendjahre_movement = {1:0, 2:1, 3:0, 4:1, 5:0, 6:1, 7:1 ,8:0 ,9:1, 10:0, 11:1, 12:0 ,13:1 ,14:0, 15:1}

        # Mappings 'Movement':
        #          Mainstream: 0 
        #          Avantgarde: 1

        # ii. Create new synthetic columns: Note that the original variable had been removed before onehot encoding

    clean_df_dum_zero["P_J_DECADE_synth"] = df_na["PRAEGENDE_JUGENDJAHRE"].map(praegende_jugendjahre_decade, na_action = "ignore")
    clean_df_dum_zero["P_J_MOVEMENT_synth"] = df_na["PRAEGENDE_JUGENDJAHRE"].map(praegende_jugendjahre_movement, na_action = "ignore")
    

    # 1.2.2.b Investigate "CAMEO_INTL_2015" and engineer two new variables.
    
        # i. Define mappings:

    CAMEO_Wealth = {'11': 1, '12': 1, '13': 1, '14': 1, '15': 1, '21': 2, '22': 2, '23': 2, '24': 2, '25': 2, '31': 3, '32': 3, '33': 3, '34': 3, '35': 3, '41': 4, '42': 4, '43': 4, '44': 4, '45': 4, '51': 5, '52': 5, '53': 5, '54': 5, '55': 5}
    CAMEO_Stage = {'11': 1, '12': 2, '13': 3, '14': 4, '15': 5, '21': 1, '22': 2, '23': 3, '24': 4, '25': 5, '31': 1, '32': 2, '33': 3, '34': 4, '35': 5, '41': 1, '42': 2, '43': 3, '44': 4, '45': 5, '51': 1, '52': 2, '53': 3, '54': 4, '55': 5}
    
        # ii. Create new synthetic columns: Note that the original variable had been removed before onehot encoding

    clean_df_dum_zero["CAMEO_Wealth_synth"] = df_na["CAMEO_INTL_2015"].map(CAMEO_Wealth, na_action = "ignore")
    clean_df_dum_zero["CAMEO_Stage_synth"] = df_na["CAMEO_INTL_2015"].map(CAMEO_Stage, na_action = "ignore")

    print('>> 1.2.2 Preprocessed data with engineered variables:')
    print('Rows: {}, Columns: {}, No. of data points: {}' .format(clean_df_dum_zero.shape[0], clean_df_dum_zero.shape[1], clean_df_dum_zero.size))
    
    preprocessed_md_count_wd_we = clean_df_dum_zero.isna().sum().sum()
    preprocessed_md_to_total_ratio_wd_we = preprocessed_md_count_wd / clean_df_dum_zero.size
    print('Missing data: Missing values: {}, No. of data points: {}, Ratio to total data points %: {}' .format(preprocessed_md_count_wd_we, clean_df_dum_zero.size, np.round(100*preprocessed_md_to_total_ratio_wd_we,2)))      

       
    return clean_df_dum_zero
    


# In[169]:


clean_demographic_data = clean_data(azdias)
# clean_demographic_data.head()


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[ ]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.


# In[ ]:


# Apply feature scaling to the general population demographics data.

scaler = StandardScaler()

clean_


# ### Discussion 2.1: Apply Feature Scaling
# 
# (Double-click this cell and replace this text with your own text, reporting your decisions regarding feature scaling.)

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[ ]:


# Apply PCA to the data.


# In[ ]:


# Investigate the variance accounted for by each principal component.


# In[ ]:


# Re-apply PCA to the data while selecting for number of components to retain.


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[ ]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.


# In[ ]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.


# In[ ]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[ ]:


# Over a number of different cluster counts...


    # run k-means clustering on the data and...
    
    
    # compute the average within-cluster distances.
    
    


# In[ ]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.


# In[ ]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.


# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[ ]:


# Load in the customer demographics data.
customers = 


# In[ ]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[ ]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.


# In[ ]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?


# In[ ]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:45:51 2019

@author: GuiReple

This script is intended to go through an exploratory analysis of the 
GOT_character_predictions.xlsx and its main objective is to predict  
which characters in the book series will live or die. 
    
The comments are designed to explain the train of thought at the moment 
and point out what is significant for the building of the various models. 

The model used in the Written Analysis is noted on lines 727-809.
The model with the best ROC AUC CV Score is noted on lines 886-981.
    
"""

##Working Directory: /Users/GuiReple/Desktop/Machine Learning

###############################################################################
################## Importing all packages #####################################
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
import statsmodels.formula.api as smf # logistic regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score    # AUC value
import seaborn as sns                        # visualizing the confusion matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

###############################################################################
############### Data Exploration (part 1) and Data Mining #####################
###############################################################################

got = pd.read_excel('~/Desktop/Machine Learning/GOT_character_predictions.xlsx')

got_df = got

print(got_df)
got_df.info()

###For-loop for missing values before data exploration. 
for col in got_df:
    
    if got_df[col].isnull().any():
        got_df['m_'+col] = got_df[col].isnull().astype(int)

###############################################################################        
### Missing Value imputation for all categorical variables. They will not be 
### used in analysis. Only the missing value flags. 
### Seems like those who have missing values, aren't really important in the
### sense of "popular" enough to have most of these informations. 
###############################################################################
        
fill = 'Unknown'
got_df['culture'] = got_df['culture'].fillna(fill)
got_df['title'] = got_df['title'].fillna(fill)
got_df['mother'] = got_df['mother'].fillna(fill)
got_df['father'] = got_df['father'].fillna(fill)
got_df['heir'] = got_df['heir'].fillna(fill)
got_df['spouse'] = got_df['spouse'].fillna(fill)
got_df['isAliveMother'] = got_df['isAliveMother'].fillna(fill)
got_df['isAliveFather'] = got_df['isAliveFather'].fillna(fill)
got_df['isAliveSpouse'] = got_df['isAliveSpouse'].fillna(fill)
got_df['isAliveHeir'] = got_df['isAliveHeir'].fillna(fill)
got_df['house'] = got_df['house'].fillna(fill)

###Simplifying the columns with book name to reduce coding space.
got_df = got_df.rename(index = str, columns ={
                                        'book1_A_Game_Of_Thrones': 'book1',
                                        'book2_A_Clash_Of_Kings': 'book2',
                                        'book3_A_Storm_Of_Swords': 'book3',
                                        'book4_A_Feast_For_Crows': 'book4',
                                        'book5_A_Dance_with_Dragons': 'book5'
                                        })
  
    
###### Age and year adjustment for super outliers that were actually 
###### uncertainity between two years.
###### Age and year of Rheago and Doreah adjusted according to research.
age_dic = {-298001:0,
           -277980:20}
got_df['age'].replace(age_dic, inplace = True)

dob_dic = {298299:298,
           278279:278}
got_df['dateOfBirth'].replace(dob_dic, inplace = True)

###### Created a culture dictionary to correct the names. 
###### The culture factor was considered in creating variables but wasn't
###### very effective in helping create variance in the data. Further 
###### consideration with these variables in the future could be benefitial. 
###### The variables were not useful in the creation of features. The data is 
###### cleaned up in culture. 

culture_count = got_df['culture']

culture_dic = {'Andal': 'Andals',
               "Asshai'i": 'Asshai',
               'Astapori':'Astapor',
               'Braavosi':'Braavos',
               'Dorne':'Dornish',
               'Dornishmen':'Dornish',
               'Free folk':'Free Folk',
               'freefolk': 'Free Folk',
               'free folk':'Free Folk',
               'Ghiscaricari':'Ghiscari',
               'Ironmen':'Ironborn',
               'Lhazarene':'Lhazareen',
               'Lyseni':'Lysene',
               'Meereenese':'Meereen',
               'Northern mountain clans':'Northmen',
               'Norvoshi':'Norvos',
               'Qartheen':'Qarth',
               'Reach':'Reachmen',
               'The Reach':'Reachmen',
               'Riverlands':'Rivermen',
               'Stormlands':'Stormlander',
               'Summer Islands':'Summer Islander',
               'Summer Isles':'Summer Islander',
               'Vale mountain clans': 'Valemen',
               'Vale':'Valemen',
               'Westerlands':'Westermen',
               'Wildling':'Free Folk',
               'Wildlings':'Free Folk',
               'ironborn':'Ironborn',
               'northmen':'Northmen',
               'westermen':'Westermen',
               'Westerman':'Westermen',
                 }

got_df['culture'].value_counts()
print(culture_count)
got_df['culture'].replace(culture_dic, inplace = True)

###############################################################################
######################## FEATURE ENGINEERING ##################################
###############################################################################

#### Culture 
#### Changing culture from object to a dummy variable so we can substitue them. 

got_df.culture = got_df.culture.astype('category')
got_df.culture.head()

dum_cult = pd.get_dummies(got_df[['culture']], prefix_sep = '_' )
        
for col in dum_cult.iloc[:, :64]:        
    culture = dum_cult[col].value_counts()
    print(culture)


##### Creating a reverse column that will flag those who are dead. The model
##### will be looking for the inverse effect where I am predicting what makes 
##### people die. 

got_df['isDead'] = got_df['isAlive'] 

##### User function to reverse binary of IsAlive.

def func(x):
    
    if x == 0:
        return 1
    else:
        return 0


got_df['isDead'] = got_df['isDead'].map(func)  

#### Creating Book related Variables. I want to add all books to see the effect
#### of appreance in books and how likely you are to die. 

got_df['hm_books'] = (got_df['book1'] + got_df['book2'] + got_df['book3'] 
                      + got_df['book4'] + got_df['book5'])
 
#### Also am creating a variable where characters were only mentioned and 
#### weren't participating in Key events in the books. 

got_df['no_books'] = got_df['hm_books']

def func(x):
    if x == 0:
        return 1
    else:
        return 0 
got_df['no_books'] = got_df['no_books'].map(func)

#### Creating combination of books flags. 

got_df['4_5_book'] = got_df['book4'] + got_df['book5']
got_df['3_5_book'] = got_df['book3'] + got_df['book5']
got_df['0124_book'] = got_df['book1'] +got_df['book2'] +got_df['book4']

#### Creating flags for people who were included in only one of the books. 

got_df['bk1_only'] = np.where((got_df.book1 == 1) & (got_df.book2 == 0) & 
                           (got_df.book3 == 0) & (got_df.book4 == 0) & 
                           (got_df.book5 == 0), "1","0")
got_df['bk1_only'].astype('int')


got_df['bk2_only'] = np.where((got_df.book2 == 1) & (got_df.book1 == 0) & 
                           (got_df.book3 == 0) & (got_df.book4 == 0) & 
                           (got_df.book5 == 0), "1","0")
got_df['bk2_only'].astype('int')


got_df['bk3_only'] = np.where((got_df.book3 == 1) & (got_df.book1 == 0) & 
                           (got_df.book2 == 0) & (got_df.book4 == 0) & 
                           (got_df.book5 == 0), "1","0")
got_df['bk3_only'].astype('int')

got_df['bk4_only'] = np.where((got_df.book4 == 1) & (got_df.book1 == 0) & 
                           (got_df.book2 == 0) & (got_df.book3 == 0) & 
                           (got_df.book5 == 0), "1","0")
got_df['bk4_only'].astype('int')

got_df['bk5_only'] = np.where((got_df.book5 == 1) & (got_df.book1 == 0) & 
                           (got_df.book2 == 0) & (got_df.book3 == 0) & 
                           (got_df.book4 == 0), "1","0")
got_df['bk5_only'].astype('int')

got_df['night_bk345'] = np.where((got_df.book5 == 1) & (got_df.book2 == 0) & 
                           (got_df.book3 == 1) & (got_df.book4 == 1) & 
                           (got_df.book1 == 0) & 
                           (got_df.house == "Night's Watch"), "1","0")

##### Creating a flag for age higher than 70. 

age_high = 70
got_df['out_age'] = 0
def conditions(got_df):
    if (got_df['age'] >= age_high):
        return 1
    elif (got_df['age'] < age_high):
        return 0

got_df['out_age'] = got_df.apply(conditions, axis=1)

##### Created a variable based on the assumption that age is either when a 
##### character is alive or the age they died in. Then subtracting by 305 (
##### the date at which the 5th book ends) will give a flag to those who have 
##### values of DOB and age and are dead or alive. 

got_df['305year_vs_dob'] = (305 - got_df['dateOfBirth'])

got_df['alive_by_age'] = 0

def conditions(got_df):
    if (got_df['age'] == got_df['305year_vs_dob']):
        return 0
    elif (got_df['age'] < got_df['305year_vs_dob']):
        return 1

got_df['alive_by_age'] = got_df.apply(conditions, axis=1)

#### Imputing missing values for the age, out_age, DOB, 300year. The reason
#### why for doing so late is that I wanted to keep the integrity of the 
#### missing values when calculating and producing new features so that after
#### it wouldn't affect the data with the imputation. -1 was chosen for missing
#### values as it stands out from the remaining observations that are real and
#### will hopefully be flagged in the model when ran together. 

got_df['age'] = got_df['age'].fillna(-1)
got_df['305year_vs_dob'] = got_df['305year_vs_dob'].fillna(-1)
got_df['dateOfBirth'] = got_df['dateOfBirth'].fillna(-1)
got_df['alive_by_age'] = got_df['alive_by_age'].fillna(-1)
got_df['out_age'] = got_df['out_age'].fillna(-1)

#### Sub-set Age and DOB without missing values in order to pursue data
#### correlation on observations that were real. 

age_noNan = got_df[got_df['age'].isin(range(1,100))]
dob_noNan = got_df[got_df['age'].isin(range(0,305))]

###############################################################################
###################### Data Exploration (part 2) ##############################
###############################################################################

sns.lmplot(x="numDeadRelations", y="hm_books", data=got_df,
          fit_reg=False,scatter=True,hue='isDead')
###############################################################################
## interesting plot that shows dead or alive hues with num dead relations in  #
## how many books they appeared. Seems that a lot of people are alive in book #
## 5 and 2 in comparisson to the proportion of people dead in 0, 1, 3, 4.     #
## The number of relatives dead seems to be increasing by the more books that #
## they are in.                                                               #
###############################################################################

 
sns.lmplot(x="age", y="hm_books", data=age_noNan,
          fit_reg=False,scatter=True,hue='isDead')
##Those who are in 0 books die a lot or were previously dead according to 
##this scatter plot.


sns.lmplot(x="age", y="numDeadRelations", data=age_noNan,
          fit_reg=False,scatter=True,hue='isDead')
###Shows distribution of dead relatives. Interesting point is that 8 and above 
### doesn't seem to hapen too much. Most of the characters with high dead (8 up)
### relatives are actually alive.  

sns.lmplot(x="age", y="isNoble", data=age_noNan,
          fit_reg=False,scatter=True,hue='isDead')
##Not the greatest plot, but shows that Noble people tend ot die a lot as the 
## hue shows a significance predominance. 

sns.lmplot(x="dateOfBirth", y="hm_books", data=dob_noNan,
          fit_reg=False,scatter=True,hue='isDead')
## Shows distribution of how "old" people are and how many books they were in
## Interesting that 0 and 1 books are older. Maybe people talk more about the 
## past in these cases. 

sns.lmplot(x="bk3_only", y="dateOfBirth" , data=dob_noNan,
          fit_reg=False,scatter=True,hue='isDead')
### There are a lot of people who only were included/ mentioned in Book 3 
### and are dead. 

sns.lmplot(x="bk4_only", y="dateOfBirth" , data=dob_noNan,
          fit_reg=False,scatter=True,hue='isDead')
### Interesting to see that in book 4 there are several characters that are
### included in only book 4 and that they are mostly alive. Something is up.
### Book 4 might be useful in feature selection. 

sns.lmplot(x="bk5_only", y="dateOfBirth" , data=dob_noNan,
          fit_reg=False,scatter=True,hue='isDead')
### This suggests that there was a split in alive and dead in people who were
### part of only book 5

sns.lmplot(x="hm_books", y="popularity", data=got_df,
          fit_reg=False,scatter=True,hue='isDead')
### The greater the participation of books, the more popular you seem to be. 
### 

sns.lmplot(x="hm_books", y="dateOfBirth", data=got_df,
          fit_reg=False,scatter=True,hue='isDead')
### This graph is great at showing the character distribution of participation 
### in the books while also showing their DOB. The over 200 seem to be 
### characters that were alive in the books and below 200 DOB seems to be 
### those important characters to the history of the plot but weren't alive
### when mentioned.  

sns.lmplot(x="night_bk345", y="popularity", data=got_df,
          fit_reg=False,scatter=True,hue='isDead')
### Night's Watch dies a lot in general. A lot during book 3,4 and 5. 


########Histograms########
plt.subplot(2, 2, 1)
sns.distplot(got_df['hm_books'],
             bins = 5,
             kde = False,
             rug = True,
             color = 'blue')
plt.xlabel('Book Distribution')
####More characters are in 2 books. The next significant number is all 5 books. 
#### Around 400 characters are in all 5 books.  

plt.subplot(2, 2, 2)
sns.distplot(got_df['isDead'],
             bins = 10,
             kde = False,
             rug = True,
             color = 'red')
plt.xlabel('Dead vs Alive')
####Distribution of is Dead. 

plt.subplot(2, 2, 3)
sns.distplot(got_df['numDeadRelations'],
             bins = 10,
             kde = False,
             rug = True,
             color = 'red')
plt.xlabel('Dead Relatives')
#### Most characters have no dead relatives.

plt.subplot(2, 2, 4)
sns.distplot(dob_noNan['dateOfBirth'],
             bins = 10,
             kde = False,
             rug = True,
             color = 'red')
plt.xlabel('DOB Distribution')
#### most of the characters who have a DOB are older and probably dead. 
#### This feature could lead to being important in a model for predicting is 
#### dead. 

plt.subplot(2, 2, 4)
sns.distplot(age_noNan['age'],
             bins = 5,
             kde = False,
             rug = True,
             color = 'red')
plt.xlabel('Age Distribution')
#### Interesting to note that age distribution is right skewed, meaning there
#### are younger people in those who have age data recorded. Could mean that 
#### A) more people are dying young. B) there are more young people than old. 


###############################################################################
########################### START OF MODELING #################################
###############################################################################

"""
I will start my modeling with a backwards in method. I want to observe which 
factorsare important and then little by little add or take away features with 
low effect in training and testing AUC. This part took some time and tweaking 
with different variables. I tried to think about what could affect survival 
rate and what wouldn't affect survival. 
"""
##### Setting up the data to run in a simple tree to identify effects of features 
##### with each other. 

got_data_simple=got_df

got_data_simple = got_df.drop(['isDead','isAlive','name','title',
                        'mother','father','heir','house',
                        'spouse','isAliveMother','isAliveFather',
                        'isAliveSpouse', 'culture','isAliveHeir'], axis = 1)

got_target_simple = got_df.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data_simple,
            got_target_simple,
            test_size = 0.10,
            stratify = got_target_simple,
            random_state = 508)


##### User Defined Function to take care of plotting for the models in future 
##### models. Will be plotting AUC feature importance. 
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')


###############################################################################
######################## Random Forests #######################################
###############################################################################

"""
Here we will be testing the two criterions of Random Forest Classifier, gini
entropy to see which one fits best with our model with all variables and see 
which ones we can replace using the feature importance graph. We will then 
create an optimal variable selection to continue running all models. 
"""

full_forest_gini = RandomForestClassifier(n_estimators = 355,
                                     criterion = 'gini',
                                     max_depth = 6,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = True,
                                     min_samples_split = 26,
                                     max_leaf_nodes = 16,
                                     oob_score = True,
                                     
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 355,
                                     criterion = 'entropy',
                                     max_depth = 6,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = True,
                                     min_samples_split = 26,
                                     max_leaf_nodes = 16,
                                     oob_score = True,
                                     
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)



# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()

############################
###Scoring Random Forest####
############################

# Scoring the gini model
print('Gini Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Gini Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Entropy Training Score', full_entropy_fit.score(X_train,
                                                       y_train).round(4))
print('Entropy Testing Score:', full_entropy_fit.score(X_test, 
                                                       y_test).round(4))

y_pred_train_treeg = full_gini_fit.predict(X_train)
y_pred_test_treeg = full_gini_fit.predict(X_test)

# Let's also check our auc value
print('Gini Training AUC Unofficial Score', roc_auc_score(y_train, 
                                                          y_pred_train_treeg).round(4))
print('Gini Testing AUC Unofficial Score:', roc_auc_score(y_test, 
                                                          y_pred_test_treeg).round(4))


y_pred_train_treee = full_entropy_fit.predict(X_train)
y_pred_test_treee = full_entropy_fit.predict(X_test)

print('Entropy Training AUC Unofficial Score', roc_auc_score(y_train, 
                                                             y_pred_train_treee).round(4))
print('Entropy Testing AUC Unofficial Score:', roc_auc_score(y_test, 
                                                             y_pred_test_treee).round(4))

random_f_gini = cross_val_score(full_forest_gini,
                           got_data_simple,
                           got_target_simple,
                           cv = 3, scoring= 'roc_auc')


print('Gini AUC Cross Validation:', random_f_gini)


print('Gini AUC CV ROC Score:', pd.np.mean(random_f_gini).round(3))

random_f_entropy = cross_val_score(full_forest_entropy,
                           got_data_simple,
                           got_target_simple,
                           cv = 3, scoring= 'roc_auc')


print('Entropy AUC Cross Validation:' , random_f_entropy)


print('Entropy AUC CV ROC Score:', pd.np.mean(random_f_entropy).round(3))

##########################
###Plotting AUC CV FIT####
##########################
plot_feature_importances(full_gini_fit,
                         train = X_train,
                         export = False)



plot_feature_importances(full_entropy_fit,
                         train = X_train,
                         export = False) 

###############################################################################
"""
Now I will be selecting the variables through trial and error to see which 
has a positive effect in the Training and Testing Score. A lot of trial and
error and plotting. This is the part where the art comes in. After some time,
I concluded the following combination that will be used throughout the 
remainder of the modeling to compare different types. 
"""

got_data = got_df
got_data = got_df.drop(['isDead','isAlive','name','title',
                        'mother','father','heir','house',
                        'isAliveMother','isAliveFather','isAliveSpouse',
                        'culture','isAliveHeir','305year_vs_dob',
                        'm_isAliveMother','m_isAliveHeir','m_mother',
                        'm_heir','m_father','m_isAliveFather',
                        'm_spouse','out_age','m_isAliveSpouse',
                        'm_house','m_dateOfBirth','book5','book2',
                        'S.No','0124_book','culture','m_culture',
                        'male','3_5_book','m_title','book1','bk4_only',
                        'bk3_only', 'bk2_only','4_5_book','no_books',
                        'm_age','age','spouse','bk3_only','book3'
                         
                        ], axis = 1)
got_target = got_df.loc[:, 'isDead']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.10,
            random_state = 508)


"""
Now that the variables were selected, I will be running a GridSearchCv to 
tune the parameter using hyperparameter methods. 
"""

###############################################################################
################ Hyperparameter tunning with Gini Vs Entropy ##################
########################## Grid Search CV #####################################
###############################################################################
"""
Arranged a RandomForrest Grid Search. The following was returned:

Tuned Logistic Regression Parameter: {'bootstrap': True, 'criterion': 
'entropy','min_samples_leaf': 1, 'n_estimators': 100, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.8601
"""

"""
# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, 
                              param_grid, 
                              cv = 3)



# Fit it to the training data
full_forest_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))  
"""

###############################################################################
##########Building Random Forest Model Based on Best Parameters################
###############################################################################

#Training Score 0.9309
#Testing Score: 0.8359
#Training AUC Score 0.8833
#Testing AUC Score: 0.739
#CrossValidation Results: [0.84943025 0.87197596 0.87534977]
#Testing CV ROC AUC Score: 0.866

rf_optimal = RandomForestClassifier(
                                    bootstrap = True,
                                    criterion = 'entropy',
                                    min_samples_leaf = 1,
                                    n_estimators = 100,
                                    warm_start = True,
                                    random_state = 508)



rf_optimal_fit = rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


rf_optimal_pred_train = rf_optimal_fit.predict(X_train)
rf_optimal_pred_test = rf_optimal_fit.predict(X_test)

print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))

# Let's also check our auc value
print('Training AUC Score', roc_auc_score(y_train, rf_optimal_pred_train).round(4))
print('Testing AUC Score:', roc_auc_score(y_test,rf_optimal_pred_test).round(4))

plot_feature_importances(rf_optimal_fit,
                         train = X_train,
                         export = False)

rf_cv_auc = cross_val_score(rf_optimal,
                           got_data,
                           got_target,
                           cv = 3, scoring= 'roc_auc')


print('CrossValidation Results:', rf_cv_auc)


print('Testing CV ROC AUC Score:', pd.np.mean(rf_cv_auc).round(3))


"""
Hyperparameter tunning seems to have increased the over-fit in the model. I 
will be switching gears to a ensembler model, GradientBoostingClassifier, to 
see if by tweaking with the learning_rate, I will be able to penalize the 
features that are causing overfiting. 

"""

###############################################################################
############### GBM Improvement from Random Forest ############################
################## Model Used in Written Analysis #############################
###############################################################################
"""
Making adjustments and adding extra parameters. Most of which I eye balled out 
to see their effects on the Training and Testing. There was a lot of trial and 
error while tweaking parameters. Most of what was done was experimenting new 
parameters that weren't explored. 
"""
#Training Score 0.8681
#Testing Score: 0.8667
#Training AUC Score 0.7819
#Testing AUC Score: 0.7597
#CrossValidation Results: [0.87136865 0.89317556 0.90342556]
#Testing CV ROC AUC Score: 0.889

gbm_optimal = GradientBoostingClassifier(
                                      criterion = 'mse',
                                      learning_rate = 1.01,
                                      max_depth = 1,
                                      n_estimators = 150,
                                      random_state = 508,
                                      min_samples_leaf = 1,
                                      #subsample = .8,
                                      warm_start = True,
                                      max_features = 'sqrt',
                                      presort = True
                                      
                                      #criterion = 'friedman_mse',
                                      #learning_rate = 1,
                                      #max_depth = 1,
                                      #n_estimators = 350,
                                      #random_state = 508,
                                      #max_features = 10,
                                      #subsample = .8,
                                      #presort = True
                                      )



gbm_fit = gbm_optimal.fit(X_train, y_train)


gbm_optimal_score = gbm_optimal.score(X_test, y_test)


gbm_optimal_pred = gbm_optimal.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))


gbm_optimal_train = gbm_optimal.score(X_train, y_train)
gmb_optimal_test  = gbm_optimal.score(X_test, y_test)

y_pred_train_tree = gbm_fit.predict(X_train)
y_pred_test_tree = gbm_fit.predict(X_test)


print('Training AUC Score', roc_auc_score(y_train, y_pred_train_tree).round(4))
print('Testing AUC Score:', roc_auc_score(y_test, y_pred_test_tree).round(4))

plot_feature_importances(gbm_fit,
                        train = X_train,
                        export = False) 

gbm_cv_auc1 = cross_val_score(gbm_optimal,
                           got_data,
                           got_target,
                           cv = 3, scoring= 'roc_auc')


print('CrossValidation Results:', gbm_cv_auc1)


print('Testing CV ROC AUC Score:', pd.np.mean(gbm_cv_auc1).round(3))

"""
This model is the one that I will use when writting the analysis.
"""

###############################################################################
###########################Logistic Regression#################################
###############################################################################

"""Running a Logistic Regression to Analyze the effects that the variables
picked have in a different model."""

#Training Score 0.8429
#Testing Score: 0.8308
#Training AUC Score 0.6992
#Testing AUC Score: 0.67
#CrossValidation Results: [0.82651515 0.84573629 0.8726959 ]
#Testing CV ROC AUC Score: 0.848

logreg = LogisticRegression(C = 0.5,
                            solver = 'lbfgs')

logreg_fit = logreg.fit(X_train, y_train)


logreg_pred = logreg_fit.predict(X_test)

print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

y_pred_train_log = logreg_fit.predict(X_train)
y_pred_test_log = logreg_fit.predict(X_test)

# Let's also check our auc value
print('Training AUC Score', roc_auc_score(y_train, y_pred_train_log).round(4))
print('Testing AUC Score:', roc_auc_score(y_test, y_pred_test_log).round(4))

logit_cross = cross_val_score(logreg,
                           got_data,
                           got_target,
                           cv = 3, scoring= 'roc_auc')


print('CrossValidation Results:', logit_cross)


print('Testing CV ROC AUC Score:', pd.np.mean(logit_cross).round(3))

###############################################################################
############################ Logistic Regression ##############################
###############################################################################
"""
Running a smf.logit regression to understand the coefficient weight of the 
features chosen for the models tested previously. There I identified the
following variables that will increase chances of survival (will be negative
coefficients since I am using IsDead): bk5_only, isMarried, book4, dateOfBirth.
By doing so I can have a better idea as to what is good and what is bad for 
survival. 
"""

log_got_p = smf.logit(formula = """isDead ~ 
                                                 got_df['alive_by_age'] 
                                                + got_df['hm_books'] 
                                                + got_df['popularity'] 
                                                + got_df['numDeadRelations'] 
                                                + got_df['bk5_only'] 
                                                + got_df['isNoble'] 
                                                + got_df['isMarried'] 
                                                + got_df['book4'] 
                                                + got_df['bk1_only'] 
                                                + got_df['dateOfBirth']""",
                                                data = got_df)


results_logistic_full = log_got_p.fit()


results_logistic_full.summary()


###############################################################################
##################### BEST MODEL IN TERMS OF AUC ##############################
###############################################################################
"""
This is the model I created with specific features that gave me the best Test
CV ROC AUC Score. I will not use this in my analysis as the story behind this 
is a bit complex to write. The books are kind of a mess, the age seems to be 
correlated with the date of Birth and the story isn't as strong as the GBM 
model mentioned previously.
"""
#Training Score 0.8601
#Testing Score: 0.8769
#Training AUC Score 0.7674
#Testing AUC Score: 0.7666
#CrossValidation Results: [0.88523666 0.88730904 0.90309932]
#Testing CV ROC AUC Score: 0.892


got_data_best = got_df
got_data_best = got_df.drop(['isDead','isAlive','name','title',
                        'mother','father','heir','house',
                        'isAliveMother','isAliveFather','isAliveSpouse',
                        'culture','isAliveHeir','305year_vs_dob',
                        'm_isAliveMother','m_isAliveHeir','m_mother',
                        'm_heir','m_father','m_isAliveFather',
                        'm_spouse','out_age','m_isAliveSpouse',
                        'm_house','m_dateOfBirth','book5','book2',
                        'S.No','book3','0124_book','culture','m_culture',
                        'male','3_5_book','m_title','book1','bk4_only',
                        'bk3_only', 'bk2_only','spouse','bk1_only', 
                        'bk5_only'
                         
                        ], axis = 1)
got_target_best = got_df.loc[:, 'isDead']

X_train, X_test_, y_train, y_test_ = train_test_split(
            got_data_best,
            got_target_best,
            test_size = 0.10,
            random_state = 508)

gbm_bestm = GradientBoostingClassifier(criterion = 'mse',
                                      learning_rate = 1.02,
                                      max_depth = 1,
                                      n_estimators = 150,
                                      random_state = 508,
                                      min_samples_leaf = 1,
                                      #subsample = .8,
                                      #warm_start = True,
                                      max_features = 'sqrt'
                                      )

gbm_best_fit = gbm_bestm.fit(X_train, y_train)


gbm_best_score = gbm_bestm.score(X_test_, y_test_)


gbm_best_pred = gbm_bestm.predict(X_test_)


# Training and Testing Scores
print('Training Score', gbm_bestm.score(X_train, 
                                        y_train).round(4))
print('Testing Score:', gbm_bestm.score(X_test_, 
                                        y_test_).round(4))


gbm_best_train = gbm_bestm.score(X_train, y_train)
gmb_best_test  = gbm_bestm.score(X_test_, y_test_)

gbm_y_best_pred_train = gbm_best_fit.predict(X_train)
gbm_y_best_pred_test = gbm_best_fit.predict(X_test_)


print('Training AUC Score', roc_auc_score(y_train, 
                                          gbm_y_best_pred_train).round(4))
print('Testing AUC Score:', roc_auc_score(y_test_, 
                                          gbm_y_best_pred_test).round(4))

auc_score = roc_auc_score(y_test_,gbm_y_best_pred_test).round(4)

plot_feature_importances(gbm_best_fit,
                         train = X_train,
                         export = False) 

gbm_best_cv_auc = cross_val_score(gbm_best_fit,
                           got_data_best,
                           got_target_best,
                           cv = 3, scoring= 'roc_auc')

print('CrossValidation Results:', gbm_best_cv_auc)


print('Testing CV ROC AUC Score:', pd.np.mean(gbm_best_cv_auc).round(3))


###############################################################################
####################### Importing Results to Excel ############################
###############################################################################

"""
Two models were included in the excel spreadsheet. I wanted to include the
model with the best ROC AUC CV score as well as the model used for the write 
up analysis. 
"""
# Storing the two model predictions as a dictionary.
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'GBM_Best_Predicted': gbm_best_pred,   
                                     'GBM_Analysis_Predicted': gbm_optimal_pred,
                                     })

# Moving predictions to excel
model_predictions_df.to_excel("GOT_Guilherme_Pred.xlsx")


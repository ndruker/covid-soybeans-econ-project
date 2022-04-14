from dataclasses import dataclass
import models as mytools
import seaborn as sns
from tabulate import tabulate
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import signal
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import PolynomialFeatures
from functools import reduce
from matplotlib.pyplot import cm, title
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap



""" HYPERPARAMETERS """
#dictionary that stores the paths to the soybean data, maps a name to file path
paths_to_soybean_data = [
        "soybean_data/mar_22.txt",
        "soybean_data/may_22.txt",
        "soybean_data/july_22.txt",
        "soybean_data/aug_22.txt",
        "soybean_data/sep_22.txt",
        "soybean_data/nov_22.txt",         
        "soybean_data/jan_23.txt",
        "soybean_data/mar_23.txt",
        "soybean_data/may_23.txt",
        "soybean_data/july_23.txt",
        "soybean_data/aug_23.txt", 
        "soybean_data/sep_23.txt",
        "soybean_data/nov_23.txt",
        "soybean_data/jan_24.txt",
        "soybean_data/mar_24.txt",
        "soybean_data/july_24.txt",
        "soybean_data/aug_24.txt",
        "soybean_data/nov_24.txt"]

# more future contracts data will be added in the future
#col names - how we parse the data
soybean_col_names = ['date', 'open','high','low','close', 'adj_close','volume']


#extract dataframes
soy_df_list = [] # store dfs in a list 
soybean_data_labels = ['March 2022',"May 2022","July 2022","August 2022","September 2022","November 2022","January 2023",
        "March 2023","May 2023","July 2023","August 2023","September 2023","November 2023","January 2024","March 2024","July 2024","August 2024","November 2024"]
#the names are ordered
for path in paths_to_soybean_data: # for each key
        df =  mytools.clean_soybean_data(path, col_names=soybean_col_names)
        # k.instrument_name(k)
        soy_df_list.append(df)


## then create a cross table with the covid data for each futures_dataset 
soy_covid_df_list = []
for df in soy_df_list:
        combined_df = mytools.make_data(df, 1) #DAILY COVID CHANGE
        soy_covid_df_list.append(combined_df)


"""
Explore the covid data as a predictor: notice that covid rates reach maturity around July 2020

"""
df_1 = soy_covid_df_list[0] ##for the sake of analysis, only use a single dataframe, for the final paper run on all future contracts datasets
my_df = df_1
df_1 = df_1[['date', 'daily_cases', 'daily_deaths', 'daily_close', 'close', 'cases_FD', 'deaths_FD', "price_FD" ]]
 

# plt.show()
df_1.plot(x='date', y='cases_FD', label='first difference of cases', title="Additional Daily Case Counts (First Difference)")
df_1.plot(x='date', y='deaths_FD', label="first difference of deaths", title="Additional Daily Death Counts (First Difference)")
plt.show()



# def draw_covid_plot(df_1):
#         # Draw Plot
#         fig, axes = plt.subplots(6, 2, figsize=(20,12), dpi= 80)
#         sns.lineplot(x='date', y='cases', data=df_1, ax=axes[0,0])
#         sns.lineplot(x='date', y='cases_pc', data=df_1, ax=axes[1,0])
#         sns.lineplot(x='date', y='cases_diff', data=df_1, ax=axes[2,0])
#         sns.lineplot(x='date', y='deaths', data=df_1, ax=axes[3,0]) 
#         sns.lineplot(x='date', y='deaths_pc', data=df_1, ax=axes[4,0])
#         sns.lineplot(x='date', y='deaths_diff', data=df_1, ax=axes[5,0])

#         #separate in July 
#         matureddf = df_1[df_1['date'] > '2020-07-01']
#         sns.lineplot(x='date', y='cases', data=matureddf, ax=axes[0,1])
#         sns.lineplot(x='date', y='cases_pc', data=matureddf, ax=axes[1,1])
#         sns.lineplot(x='date', y='cases_diff', data=matureddf, ax=axes[2,1])
#         sns.lineplot(x='date', y='deaths', data=matureddf, ax=axes[3,1]) 
#         sns.lineplot(x='date', y='deaths_pc', data=matureddf, ax=axes[4,1]) 
#         sns.lineplot(x='date', y='deaths_diff', data=matureddf, ax=axes[5,1]) 

#         # Set Title
#         fig.suptitle("Covid rates became stable in July 2020", fontsize=18)
#         axes[0,0].set_title('January to current', fontsize=18); 
#         axes[0,1].set_title('July to current', fontsize=18)
#         plt.show()

# #plot that shows a pivot in july 2020 - covid rate become stable
# # draw_covid_plot(df_1)


"""Plot the seasonality of cases and deaths"""
def plot_seasonality(df, y, ax0title, ax1title):
        # Prepare data
        df['year'] = [d.year for d in df.date]
        df['month'] = [d.strftime('%b') for d in df.date]
        # years = df['year'].unique()

        # Draw Plot
        sns.set(font_scale = 1.2)
        fig, axes = plt.subplots(1, 2, figsize=(16,8), dpi= 80)
        sns.boxplot(x='year', y=y, data=df, ax=axes[0])
        sns.boxplot(x='month', y=y, data=df.loc[~df.year.isin([2019, 2024]), :])

        # Set Title
        axes[0].set_title(ax0title, fontsize=14); 
        axes[1].set_title(ax1title, fontsize=14)
        plt.show()

plot_seasonality(df_1, 'cases_FD', 'Daily Additional Cases by Year\n (trend)', 'Daily Additional Cases by Month\n (seasonality)' )
plot_seasonality(df_1, 'deaths_FD', 'Daily Additional Deaths by Year\n (trend)', 'Daily Additional Deaths by Month\n (seasonality)' )



""""Test if the predictor is stationary- makes for more accurate linear predictions"""
# cases is stationary while trends is non-stationary! null is non-stationary. we want to reject the null!
stationary_metrics = []
for metric in list(["cases_FD", 'deaths_FD']):
        for i in range(len(soy_covid_df_list)):
                table = soy_covid_df_list[i][metric].dropna(axis=0)
                result = adfuller(table.values, autolag='AIC')
                print(f'ADF Statistic: {result[0]}', "for ", metric, " taken from ", soybean_data_labels[i],"futures data ")
                print(f'p-value: {result[1]}')
#                 if result[1] < 0.05:
#                         stationary_metrics.append(metric)
#                 for key, value in result[4].items():
#                         print('Critial Values:')
#                         print(f'   {key}, {value}')

stationary_metrics = list(set(stationary_metrics))

"""Autocorrelation test"""
# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(df_1.value, nlags=50)
# pacf_50 = pacf(df_1.value, nlags=50)

# Draw Plot
"""We see high autocorrelation in the data for the metrics that are non stationary"""
for m in stationary_metrics:
        fig, axes = plt.subplots(1,2,figsize=(12,4), dpi= 100)
        fig.suptitle('Covid ' + m + ' autocorrelation and partial autocorrelation \n 40 lagged terms', fontsize=14)
        plot_acf(df_1[m].tolist(), lags=40, ax=axes[0])
        plot_pacf(df_1[m].tolist(), method='ywm', lags=50, ax=axes[1])
        fig.tight_layout()
        plt.show()



# """PLOT lags"""
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Plot
fig, axes = plt.subplots(4, 3, figsize=(9,10), dpi=100)
for i, ax in enumerate(axes[:4, 0]):
    lag_plot(df_1.cases_FD, lag=i+1, ax=ax, c='blue')
    ax.set_title('Additional Daily Cases (FD) ' + str(i+1))
for i, ax in enumerate(axes[:4, 1]):
    lag_plot(df_1.deaths_FD, lag=i+1, ax=ax, c='red')
    ax.set_title('Additional Daily Deaths (FD) ' + str(i+1))
for i, ax in enumerate(axes[:4, 2]):
    lag_plot(df_1.close, lag=i+1, ax=ax, c='orange')
    ax.set_title('Daily Close Price ' + str(i+1))
 
fig.tight_layout()
plt.show()



"""Granger Causality """
df_1 = df_1.dropna(axis=0)
# Determine the number of lags using in the model with this function
for m in ["deaths_FD", "cases_FD"]:
        print("__________________" + m + "__________________")
        gct = grangercausalitytests(df_1[[m, 'close']], maxlag=2) ##adjust max lags based on the plot 






"""
MODEL: 
        OLS mapping the affect of additional covid cases and deaths on closing prices of Soybean futures
"""

all_models_parameters = []
all_models_rsq = []
all_models_mse = []
poly_models_parameters = []
poly_models_rsq = []
poly_models_mse = []

for dataframe, label in zip(soy_covid_df_list, soybean_data_labels):
        #set up independent variables 
        X=dataframe.loc[:,['cases_FD','deaths_FD']]
        # # X['price_lag_2'] = df_1['close'].shift(periods= -2) 
        X['cases_lag'] = X['cases_FD'].shift(periods= -1) 
        X['deaths_lag'] = X['deaths_FD'].shift(periods= -1) 
        X['price_lag'] = dataframe['close'].shift(periods= -1) 
        X = sm.add_constant(X) # add a bias term        

        X.fillna(X.mean(), inplace=True)

        #set dependent variable
        y = dataframe.loc[:,'close'] # labels
        # initialize model 
        model = sm.OLS(y,X, missing='drop')
        #train model
        result = model.fit()
        #test model performance
        predicttions = result.predict(X)
        all_models_mse.append(result.mse_total)
        all_models_rsq.append(result.rsquared)
        # plt.scatter(df_1.date, df_1.close,  color='red')
        # plt.plot(df_1.date, predicttions, color='blue')
        # plt.legend()
        # plt.show()
        # print(label)
        # print(result.summary())
        # print(result.pvalues)
        all_models_parameters.append(result.params)

        ## Polynomial expansion ##
        X.fillna(X.mean(), inplace=True)
        #define polynomial features
        polynomial_features= PolynomialFeatures(degree=2)
        xp = polynomial_features.fit_transform(X)
        # print(xp.shape)
        #initialize and train the model
        poly_model = sm.OLS(y, xp).fit()
        poly_ypred = poly_model.predict(xp) 
        # print("X and Y shapes")
        # print(X.shape)
        # print(y.shape)
        # print("predictions shape ", poly_ypred.shape)
        poly_models_mse.append(poly_model.mse_total)
        poly_models_rsq.append(poly_model.rsquared)
        poly_models_parameters.append(poly_model.params)

        # #plot
        # plt.scatter(df_1.date, df_1.close,  color='red')
        # plt.plot(df_1.date, poly_ypred, color='blue')
        # plt.legend()
        # plt.show()
        # print(poly_model.summary())
        
#save result table
# resultstable = [soybean_data_labels, all_models_rsq, all_models_mse]
# content2=tabulate(resultstable)
# text_file=open("table.csv","w")
# text_file.write(content2)
# text_file.close()


##### FINDINGS #####

results_df = pd.DataFrame(data={ 'maturity':soybean_data_labels, 'cases_coef':[i[1] for i in all_models_parameters],
 'deaths_coef':[i[2] for i in all_models_parameters]})

plt.rcParams.update({'ytick.left': True, 'axes.titlepad':10})
fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=False, dpi= 90)

c = sns.lineplot(x='maturity', y='cases_coef', data=results_df, ax=axes[0], err_style='band', ci='std', label='Covid-19 cases coefficient')
d = sns.lineplot(x='maturity',y='deaths_coef', data=results_df, ax=axes[1], err_style='band', ci='std',label='Covid-19 deaths coefficient')
c.set_xlabel("Contract Maturity", fontsize = 15)
c.set_ylabel("Coefficient on additional daily cases", fontsize = 15)
sns.regplot(x=results_df.index, y="cases_coef", data=results_df, ax=axes[0] )
sns.regplot(x=results_df.index, y="deaths_coef", data=results_df, ax=axes[1])

d.set_xlabel("Contract Maturity", fontsize = 15)
d.set_ylabel("Coefficient on additional daily deaths", fontsize = 15) 
c.axhline(-.0000007898850, c='red', label="PC1 coefficient")
d.axhline(0.0001639670, c='red', label="PC1 coefficient")
c.legend()
d.legend()
 

# Set Title
axes[0].set_title('Coefficient of Daily Cases (First Difference)', fontsize=12); 
axes[0].tick_params(labelrotation= -55)

# plt.xticks(rotation=30)
axes[1].set_title('Coefficient of Daily Deaths (First Difference)', fontsize=12); 
fig.suptitle('Coefficients for Detrended Daily Levels of Covid-19 Cases and Deaths Across Different Maturities between 2022-2024', fontsize=14)
# plt.xticks(rotation=45)
axes[1].tick_params(labelrotation=-55)

plt.tight_layout()

# plt.xticks(rotation=30) 
plt.show() 


pca_aggregate = []
for df in soy_covid_df_list:#[:14]:
        pca_aggregate.append(df[['date', 'close']])

df_merged = reduce(lambda left,right: pd.merge(left,right,on=['date'],
                                            how='outer'), pca_aggregate)

df_merged.columns = ['date'] + soybean_data_labels#[:14]
df_merged.set_index('date', inplace=True)
# print(df_merged)

df_merged = df_merged.dropna(axis=0)


"""
PCA: dimensionality reduction

"""



X = df_merged.values # getting all values as a matrix of dataframe 

X_means_subtracted = X.mean(axis=1).mean()
sc = StandardScaler() # creating a StandardScaler object
X_std =  X#sc.fit(X).transform(X) # standardizing the data

print("mean subtracted:" ,X_means_subtracted)

#initialize the object
pca = PCA()
X_pca = pca.fit(X_std)

plt.plot(np.arange(18), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.xticks(np.arange(18))
plt.axhline(y=0.99629, color='r', linestyle='--')
plt.ylabel('cumulative explained variance')
plt.title("Explained Variance by Number of Principal Components Found By PCA")
plt.show()



## 
num_components = 1
pca = PCA(num_components)  
X_pca = pca.fit_transform(X_std) # fit and reduce dimension

# print(df_merged.indexy_rescaled)
view = pd.DataFrame(pca.components_, columns = df_merged.columns)



# loading_scores = pd.Series(columns=pca.components_[0], index = df.gene)

loading_scores = pd.Series(pca.components_[0], index = df_merged.columns)
# Print loading matrix
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
 
# print(sorted_loading_scores)


fig, axes = plt.subplots(1,2)
color = cm.rainbow(np.linspace(0, 1, len(soybean_data_labels)))
for i, c in zip(range(14), color):
        # axes[0].plot(df_merged.index, X[:,i], c=c)
        axes[0].scatter(df_merged.index, X[:,i], c=df_merged.index)

axes[0].legend()
axes[0].tick_params(labelrotation=-55)

# axes[0].scatter(df_merged.index, list(X[:,2]), c='orange')
# axes[0].scatter(df_merged.index, list(X[:,3]), c='yellow')
# axes[0].scatter(df_merged.index, list(X[:,4]), c='magenta')
# axes[0].scatter(df_merged.index, list(X[:,5]), c='purple')
# axes[0].scatter(df_merged.index, list(X[:,6]), c='black')

axes[0].set_xlabel('Date')
axes[0].set_ylabel('Closing Price')
axes[0].set_title('Closing prices at different maturities')


axes[1].scatter(df_merged.index, list(X_pca[:,0]), c=df_merged.index)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('PC1')
axes[1].set_title('PC of closing price')
axes[1].tick_params(labelrotation=-55)
plt.tight_layout()
plt.show()



#### Regression on PC1 ####
cov_df = soy_covid_df_list[0]
cov_df = cov_df[['date', 'cases_FD', 'deaths_FD', 'close']]
pca_df = pd.DataFrame(data=X_pca, index= df_merged.index)
pca_df.reset_index(inplace=True)
pca_df.columns = ['date', 'PC1']

pc1_df = cov_df.merge(pca_df, how='inner', on='date')
X = pc1_df[['cases_FD', 'deaths_FD']]


# # X['price_lag_2'] = df_1['close'].shift(periods= -2) 
X['cases_lag'] = X['cases_FD'].shift(periods= -1) 
X['deaths_lag'] = X['deaths_FD'].shift(periods= -1) 
X['price_lag'] = pc1_df['PC1'].shift(periods= -1) 
X = sm.add_constant(X) # add a bias term        

X.fillna(X.mean(), inplace=True)
#set dependent variable
y = pc1_df.loc[:,'PC1'] # labels 


# y_scaled.min(axis=0) 
# initialize model 
model = sm.OLS(y, X, missing='drop')
#train model
result = model.fit()
#test model performance
predictions = result.predict(X)
plt.title("OLS Regression on Principal Componet of Closing Price at Different Maturities between 2022-2024")
plt.xlabel("date")
plt.ylabel("PC1")
plt.xticks(rotation=-55)

plt.scatter(pc1_df.date, y,  color='red', label="actual")
plt.plot(pc1_df.date, predictions, color='blue', label="prediciton")
plt.legend()
plt.tight_layout
plt.show()

print(result.summary())
print(result.params)
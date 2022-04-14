ECON1680: Estimating The Disruptive Impact of COVID-19 Infections and Deaths on The American Soybean Industry
===============

## OVERVIEW
This repository contains the code used in my project to explain the correlation between covid-19 infections in the U.S. and the closing price of Soybean Futures maturing between March 2022 and November 2024.

## RESEARCH QUESTION
Soybeans account for 90 percent of U.S. oilseed production and 20 percent of total agricultural exports. The soybean market measures \$115.8 billion dollars per year (2020), constituting 0.65 percent of the nation's gross domestic product (GDP) and up to nine percent of the GDP for certain states. In 2020, the value of U.S. soybean exports increased 40 percent by value from the prior year, mostly due to increasing deals with China. The myriad of domestic and industrial applications of soybean supports a staggering growth rate of 4.1 percent per year (2017-2022), fueled by technological developments that increase crop yields. American soybean futures are traded on the Chicago Board of Trade (CBOT) with different maturities corresponding to growing cycles. Futures pricing controls for a variety of external factors, such as supply and demand conditions and climate. This paper attempts to quantify the effects of the economic disruption caused by the Covid-19 pandemic on the American soybean industry, which is composed of over 500,000 farmers. I explore how changes in Covid-19 rates are associated with the closing price of soybean futures at different maturities by examining the trend of correlation coefficients for daily additional cases and deaths as maturities extend into the future. I then compare my findings to the corresponding coefficients of a regression over an overall price attribute derived by means of dimensionality reduction. The results reflect an estimate of the scale of the adverse economic impact of the pandemic on the national soybean industry and also provide an outlook at how we might expect this impact to mitigate over time. 


## DATA
I combined two datasets to analyze the impact of Covid-19 rates on soybean futures: the New York Times (NYT) national cases and deaths data, and Yahoo Finance reporting of 18 active soybean futures traded on the CBOT with maturities between March 2022 and November 2024. The NYT’s extensive Covid-19 data dates back to January 2020 and records aggregate levels. I transformed the aggregate data into daily levels and then took the first difference of both attributes to define the daily change in cases and deaths levels. Yahoo Finance reports future contracts’ prices’ open, close, high, low, and volume for each day of trading. For the regression analysis, I model the daily change in cases and deaths levels against the closing price of a soybean future at a given day — as by the efficient market hypothesis, the price should reflect new information about the market, including the Covid-19 levels for that day (published on the NYT website). There are 560 daily datapoints for futures with 2022 maturities, 340 daily datapoints for futures with 2023 maturities, and 89 daily datapoints for futures with 2024 maturities.


## USAGE
the models.py file contains functions used for analysis in the analysis.py script. Not all functions have been migrated to the models.py file.

## METHOD

OLS, Polynomial expansion using ordinary least squares, PCA to determine overall price variation across maturities


## RESULTS
The results of my analysis are enclosed in the paper <link>

The results directory contains visualizations produced from the data in throughout the analysis proccess.  

## REPLICATION INSTRUCTIONS
run _$Python ./analysis.py_ to reproduce plots, charts, and regression models shown in the results section of the paper linked above.


## RESOURCES
nyt covid repo: 
nytimes. 2022. “Nytimes/Covid-19-Data: An Ongoing Repository of Data on Coronavirus Cases and Deaths in the U.S.” GitHub. March 14, 2022. https://github.com/nytimes/covid-19-data.


futures datasource:
“Soybean Futures,Jul-2022 (ZSN22.CBT) Stock Historical Prices & Data - Yahoo Finance.” 2022. @YahooFinance. 2022. https://finance.yahoo.com/quote/ZSN22.CBT/history?period1=1577836800&period2=1646092800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true.


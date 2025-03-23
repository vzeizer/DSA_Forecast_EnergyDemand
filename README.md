# Project with Feedback Number 8: Data Science Academy Data Scientist Bootcamp
## Predictive Modeling in IoT - Energy Usage Forecasting

This IoT project aims to create predictive models for forecasting the energy consumption of appliances. The data used includes measurements from temperature and humidity sensors in a wireless network, weather forecasts from an airport station, and energy usage by light fixtures.

In this machine learning project, you should perform data filtering to remove non-predictive parameters and select the best features for prediction. The dataset was collected over 10-minute periods for about 5 months. The temperature and humidity conditions of the house were monitored with a ZigBee wireless sensor network.

Each wireless node transmitted temperature and humidity conditions approximately every 3 minutes. Then, the average of the data was calculated for 10-minute periods. Energy data was recorded every 10 minutes with m-bus energy meters. The weather from the nearest airport weather station **(Chievres Airport, Belgium)** was downloaded from a public dataset from Reliable Prognosis **(rp5.ru)** and merged with the experimental datasets using the date and time column. Two random variables were included in the dataset to test regression models and filter non-predictive attributes (parameters).

Your task now is to build a predictive model that can forecast energy consumption based on the collected IoT sensor data. We recommend using **RandomForest** algorithm for feature selection and **SVM, Multilinear Logistic Regression, or Gradient Boosting** for the predictive model. The training and test datasets are attached to this PDF.


### Data Dictionary

1. `date`: Date and time of measurement (recorded every 10 minutes)

2. `Appliances`: Energy consumption of appliances in Wh (Watt-hour)

3. `lights`: Energy consumption of light fixtures in Wh (Watt-hour)

4. Temperature and humidity sensors from ZigBee network (10-minute average):
   - `T1` to `T9`: Temperature at different points in the house (°C)
   - `RH_1` to `RH_9`: Relative humidity corresponding to each temperature sensor (%)

5. Weather data from Chievres Airport:
   - `T_out`: Outside temperature (°C)
   - `Press_mm_hg`: Atmospheric pressure (mm of mercury)
   - `RH_out`: Outside relative humidity (%)
   - `Windspeed`: Wind speed (likely in m/s)
   - `Visibility`: Visibility (likely in km)
   - `Tdewpoint`: Dew point temperature (°C)

6. Random variables (included for model testing):
   - `rv1`: Random variable 1
   - `rv2`: Random variable 2

7. Time-related variables:
   - `NSM`: Number of minutes since midnight
   - `WeekStatus`: Indicates whether it's a workday or weekend
   - `Day_of_week`: Day of the week (1-7)

Important notes:
- Temperature and humidity data (T1-T9, RH_1-RH_9) are collected approximately every 3 minutes and then averaged for 10-minute periods
- Energy consumption (Appliances and lights) is recorded directly every 10 minutes
- Variables rv1 and rv2 should be discarded in the analysis as they are non-predictive by definition
- Weather data comes from an external source (rp5.ru) and was merged using the date/time


## Results

### Unique values for each Feature

The following figure shows the number of unique values for each feature available in the dataset. It can be noticed that "date", "rv2", and "rv1" present all different values, and "Appliances", the *target variable*, does not show a wide range of unique values, suggesting it displays discrete values.

![logo](images/1_uniquevalues.png)

### Negative values

By inspecting the dataset, there have been found three features which displayed negative values in the dataset, as shown in the figure below. These negative values could be potential issues when measuring those values.

![logo](images/2_hist_negativecols.png)

### Histogram of all features

The next figure shows histograms of the features available in the dataset, which interestingly points out that *Appliances* and *lights* present discrete values accross the dataset. As well, the other features's distributions can also be visually inspected. 

![logo](images/3_hist_allfeatures.png)

### Histogram for the first instance of *lights*

![logo](images/4_1_histlight1.png)

### Histogram for the second instance of *lights*

![logo](images/4_2_histlight2.png)

### Histogram for the third instance of *lights*

![logo](images/4_3_histlight3.png)

### Histogram for the fourth instance of *lights*

![logo](images/4_4_histlight4.png)

### Histogram for the fifth instance of *lights*

![logo](images/4_5_histlight5.png)

### Histogram for the sixth instance of *lights*

![logo](images/4_6_histlight6.png)

### Violinplot of all features

Next, it is shown a figure displaying violin plots for all the features in the dataset.

![logo](images/5_violinplot_features.png)

### Boxplot of all features

Next, it is shown a figure displaying box plots for all the features in the dataset.

![logo](images/6_boxplot_features.png)

### Feature Engineering

### Appliances shifted by one week

In the following figure, it is shown a visual comparison between the actual *Appliances* measure against its *7-day lagged version*.

![logo](images/7_appliances_lag7.png)

### Shifted rolling mean of Appliances

In the following figure, it is shown a visual comparison between the actual *Appliances* measure against its *one-day shifted 7 days rolling mean* version.

![logo](images/8_appliances_lag1_rollmean7.png)

### Exponential-weighted Moving Average of Appliances 

In the following figure, it is shown a visual comparison between the actual *Appliances* measure against its *one-day shifted exponential moving-average* version.


![logo](images/9_appliances_ewm1.png)

### Feature's correlation matrix

Next, it is shown the correlation matrix of the features present in the dataset, by considering the *spearman correlation method*, since this data is not normally distributed. Regardless of "Appliances", and "Appliances_std", it can be noticed that the features "T3_std", "T3_iqr", "RH_3_std", "RH_1_std" are highly correlated with "Appliances", the target variable. Therefore, these previously mentioned features are expect to contribute heavily for the machine learning model output.

![logo](images/10_corr_spearman.png)

### Machine Learning Modeling

The next figure shows the graphical result of the **SVR** (Support Vector Regressor) model in the testing data, evidencing a good agreement between prediction and the testing target variable, which is corroborated by the R-squared value in the testing set equals to 0.6. 

![logo](images/11_svrmodel_test_pred.png)

### Feature's Importance

The following figure shows the feature's relative importance/contribution in the model's output by considering the **Permutation Feature Importance**, since the usual implementation of the feature's importance method in sk-learn can not be used in Support Vector related methods. 

#### Core Idea of Permutation Feature's Importance

- The method measures how much a model's performance decreases when a specific feature's values are randomly shuffled (permuted).
**If shuffling a feature significantly reduces the model's accuracy (or another relevant metric), that feature is considered important.**
Conversely, if shuffling a feature has little to no impact on performance, that feature is deemed less important.
- Steps Involved:

1. Train the Model:
First, you train your SVM (or any other model) on your training data.
2. Establish Baseline Performance:
Evaluate the model's performance on a validation or test set, recording the baseline score (e.g., accuracy, AUC).
3. Permute Features:
For each feature, one at a time:
Randomly shuffle the values of that feature in the validation/test set.
Make predictions using the modified dataset.
Calculate the model's performance score with the shuffled feature.
Compare the new score to the baseline score.
4. Calculate Importance:
- The difference between the baseline score and the score with the shuffled feature represents the feature's importance.
- Larger differences indicate higher importance.
5. Repeat and Average:
To reduce randomness, the permutation process is often repeated multiple times, and the results are averaged.

- **Key Considerations for SVM**:

- Non-linear SVMs:
Permutation importance is particularly useful for non-linear SVMs (e.g., those using RBF kernels), where feature importance cannot be directly derived from model coefficients.
- Model-Agnostic:
A significant advantage of permutation importance is that it's model-agnostic, meaning it can be applied to any trained machine learning model.
- Correlation:
If features are highly correlated, permutation importance can produce misleading results. Shuffling one feature might not significantly impact performance if another correlated feature provides similar information.
Scoring Metric:
The choice of scoring metric (e.g., accuracy, precision, recall, AUC) can influence the results.


![logo](images/12_permutationimportance_svr.png)


## Summary of the Key Findings



## Real-world Scenario Usage





## Acknowledgements




## MIT License

Copyright (c) [2025] [Dr. Vagner Zeizer Carvalho Paes]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

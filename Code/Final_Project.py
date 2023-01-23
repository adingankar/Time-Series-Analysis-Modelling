import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from Toolbox import *
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from scipy import signal
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

#6- Description of the dataset. Describe the independent variable(s) and dependent variable:
#a. Pre-processing dataset: Dataset cleaning for missing observation. You must follow the
#data cleaning techniques for time series dataset.
#b. Plot of the dependent variable versus time.
#c. ACF/PACF of the dependent variable.
#d. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient.
#e. Split the dataset into train set (80%) and test set (20%).

#Pre-processing dataset: Dataset cleaning for missing observation. You must follow the data cleaning techniques for time series dataset."
# Loading and reading the dataset into a dataframe:
df = pd.read_csv('AirQualityBeijing.csv')

#printing first ten rows of the dataset
print(df.head(10))

#printing the info of the dataset
print(df.info())

##since the data is having missing values over the period starting from 2010 - 2013 hence we will focusing only for the year 2014.
data = df[35064:43824]

#printing the first 5 rows of the data to see.
print(data.head())

#checking the dependant column pm2.5 for null values.
null_pm = data['pm2.5'].isna().sum()
print(null_pm)

#We will be imputing the missing values by using mean of the dependant column.
mean_pm = np.mean(data['pm2.5'])
print(mean_pm)
data['pm2.5']= data['pm2.5'].fillna(mean_pm)
print(data['pm2.5'])
#Reforming the time column in the specific format yyyy:mm:dd:h:m:s
data['Time'] = data.apply(lambda x : datetime.datetime(year=x['year'], month=x['month'], day=x['day'], hour=x['hour']), axis=1)
data.drop(columns=['year', 'month', 'day', 'hour', 'No'], inplace=True)
data.time = pd.to_datetime(data.Time)
data = data.set_index('Time')
data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
print(data.head())

#Validating the unique values of the column cbwd
unique_cbwd = data['wnd_dir'].unique()
print(unique_cbwd)
#Dropping the column cbwd:
del data['wnd_dir']
print(data.head())

#b. Plot of the dependent variable versus time.
data['pollution'].plot()
plt.xlabel('Time(Daily)')
plt.ylabel('Pollution Concentration')
plt.title('Plot Graph for Pollution(pm2.5) vs TimeStep')
plt.grid()
plt.legend('pollution')
plt.figure(figsize=(20,20))
plt.show()

#c. ACF/PACF of the dependent variable.
dependant_variable = data['pollution']
mean_dep = np.mean(dependant_variable)
var = dependant_variable - mean_dep
ACF_PACF_Plot(dependant_variable,50)
ACF_PACF_Plot(dependant_variable,400)

#d. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient.
#Plotting heat map:
corr_data = data.corr()
sns.heatmap(corr_data , annot=True, cmap="Blues")
plt.title("HeatMap for Air Quality Dataset")
plt.show()


#e. Split the dataset into train set (80%) and test set (20%).
data_train , data_test = train_test_split(data , shuffle=False , test_size=0.2)
X = data[['temp', 'press','wnd_spd', 'snow',  'rain']]
Y = data['pollution']
#X_svd = sm.add_constant(X)
X_train , X_test , y_train , y_test = train_test_split(X, Y , shuffle=False, test_size=0.2)


#7- Stationarity: Check for a need to make the dependent variable stationary. If the dependent
#variable is not stationary, you need to use the techniques discussed in class to make it stationary.
#Perform ACF/PACF analysis for stationarity. You need to perform ADF-test & kpss-test and plot
#the rolling mean and variance for the raw data and the transformed data.

#Plotting the Rolling Mean and Variance for pm2.5:
rollingmean_pm = data['pollution']
cal_rolling_mean_var(rollingmean_pm , data.index)

##performing the ADF-cal and KPSS test to check the stationarity of the dependant variable.
ADF_Cal(dependant_variable)
kpss_test(dependant_variable)

#Observation: Since the data is stationary hence we will proceed further with the steps.
#using differencing technique

log_transformed_data = np.log(data['pollution'])
data_diff1 = log_transformed_data.diff()[1:]
cal_rolling_mean_var(data_diff1 , range(len(data_diff1)))

#8- Time series Decomposition: Approximate the trend and the seasonality and plot the detrended
#and the seasonally adjusted data set. Find the out the strength of the trend and seasonality. Refer
#to the lecture notes for different type of time series decomposition techniques.

from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
temp = data['pollution']
#df = pd.Series(np.array(data['pollution']), index = pd.date_range(start = '2014-01-01', periods =len(data['pollution']), freq='b'), name= 'Pollution Concentration plot')
STL = STL(temp)
res = STL.fit()
res.plot()
plt.show()

#Calculating the Trend , resid and seasonal by plotting a graph.
T = res.trend
R = res.resid
S = res.seasonal
plt.plot(T , label = 'Trend')
plt.plot(R , label = 'Resid')
plt.plot(S , label = 'Seasonal')
plt.title("Trend-Seasonality-Residuals for Pollution(pm2.5)")
plt.xlabel('Time(Daily)')
plt.ylabel('Pollution')
plt.legend()
plt.tight_layout()
plt.show()

#calculating the strength of the trend of data:
import numpy as np
v=1 - (np.var(R)/np.var(T + R))
strength_trend = np.max([0,v])
print("The strength of trend for this data set is :", strength_trend)

#calculating the strength of seasonality of the data:
v2=1 - (np.var(R)/np.var(S + R))
strength_seasonal = np.max([0,v2])
print("The strength of seasonality for this data set is :",strength_seasonal)

#Observation: Since the strength of trend is 0.88 which is close to 1 hence we can say that the data is trended and so will detrend the data and plot the graph against time to see the observations made.
#calculating the detrended data and plot is vs original data.
detrended= data['pollution'] - T
print("The strength of seasonality for this data set is :",detrended)
fig = T.plot(label = 'Detrended Data')
fig = temp.plot(label = 'Original Data')
plt.title("Plot for Detrended data vs Original data")
fig.legend()
plt.ylabel('Pollution')
plt.show()

#calculating the seasonally adjusted data plot  vs original data.
seasonality = data['pollution'] - S
print("The strength of seasonality for this data set is :",seasonality)
fig = S.plot(label = 'Seasonal Data')
fig = temp.plot(label = 'Original Data')
plt.title("Plot for Seasonally Adjusted data vs original data")
plt.ylabel('Pollution')
fig.legend()
plt.show()


### 8. Using the Holt-Winters method try to find the best fit

train_HLWM = ets.ExponentialSmoothing(data['pollution'],trend='mul',damped_trend=True,seasonal='mul').fit()
HLWM_prediction_train = train_HLWM.forecast(steps=len(data_train['pollution']))
test_HLWM = train_HLWM.forecast(steps=len(data_test['pollution']))
test_predict_HLWM = pd.DataFrame(test_HLWM).set_index(data_test['pollution'].index)
resid_HLWM = np.subtract(y_train.values,np.array(HLWM_prediction_train))
forecast_error_HLWM = np.subtract(y_test.values,np.array(test_HLWM))
MSE_HLWM = np.square(resid_HLWM).mean()
print("Mean square error for (training set) HLWM is ", MSE_HLWM)
acf_resid = auto_correlation(resid_HLWM)
#plotting the acf plot for holts winter:
var1 =np.arange(0,20)
m=1.96/np.sqrt(len(data.pollution))
plt.stem(var1,acf_resid,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*var1,acf_resid,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Holts Winter Method')
plt.axhspan(-m,m,alpha = .1, color = 'green')
plt.tight_layout()
plt.show()
Q  = (len(resid_HLWM)) *np.sum(np.square(acf_resid))
print("\nThe Q value of residual using HWM is  ",Q)
print(f"The Mean of residual of HLWM is {np.mean(resid_HLWM)}")
print(f"The variance of residual of HLWM is {np.var(resid_HLWM)}")
MSE = np.square(np.subtract(data_test['pollution'].values,np.ndarray.flatten(test_HLWM.values))).mean()
print("Mean square error for Holt Winter method is of testing set is ", MSE)
print(f"The Mean of forecast of HLWM is {np.mean(forecast_error_HLWM)}")
print(f"The variance of forecast of HLWM is {np.var(forecast_error_HLWM)}")
print(f"\n The ratio of resid vs forecast is {(np.var(resid_HLWM)) / (  np.var(forecast_error_HLWM)  )}")
plt.plot(data_train['pollution'],label= "AirQuality-train")
plt.plot(data_test['pollution'],label= "AirQuality-test")
plt.plot(test_predict_HLWM,label= "Holt-Winter Method-test")
plt.legend(loc='upper left')
plt.title('Holt-Winter Method for Pollution(pm2.5) Prediction')
plt.xlabel('Time(Daily)')
plt.ylabel('Pollution')
plt.tight_layout()
plt.show()


#Feature Selection:
from numpy import linalg as LA
X_m = X_train.values
y_m = y_train.values
x_svd = sm.add_constant(X_m)
H_vector = np.matmul(x_svd.T, x_svd)
s, d, v = np.linalg.svd(H_vector)
print(f"SingularValues = {d}")
print(f"The condition number constant (original data) = {LA.cond(x_svd)}")

#Feature Selection - OLS:
X_train=sm.add_constant(X_train)
model = sm.OLS(y_train , X_train).fit()
X_test = sm.add_constant(X_test)
predictions =model.predict(X_test)
print(model.summary())

#We will be dropping the feature Is:
X_train.drop('snow',axis = 1, inplace=True)
model_1 = sm.OLS(y_train , X_train).fit()
X_test.drop('snow', axis = 1 ,inplace=True)
predictions_1 = model_1.predict(X_test)
print(model_1.summary())

#Will be dropping the constant to check the model summary:
X_train.drop('const',axis = 1, inplace=True)
model_final = sm.OLS(y_train , X_train).fit()
X_test.drop('const', axis=1 , inplace=True)
predictions_2 = model_final.predict(X_test)
print(model_final.summary())

prediction_test = predictions_2
# predictions_2 is the final prediction for x test based on my multiple regression model
#so I will name it as prediction_test

#Multiple Linear Regression:
#performing the f-test on the model obtained after backward-stepwise regression.
f_test = model_final.fvalue
print('\nF-statistic: ', f_test)
print("Probability of observing value at least as high as F-statistic ",model_final.f_pvalue)

#Performing the t-test on the model obtained after backward-stepwise regression.
t_Test =model_final.pvalues
print("\nT-test P values : ", t_Test)

#now I need to predict based on train set so.
model_train = sm.OLS(y_train, X_train).fit()
prediction_train =model_train.predict(X_train)
training_residual = np.subtract(y_train,prediction_train)

MSE = np.square(training_residual).mean()
print("Mean square error of training set for multiple regression is ", MSE)
print("RMSE for training set using multiple regression is :",MSE)

def calc_Q_value(x):
    #calc_autocorr,calc_autocorr_np = auto_corelation(x, statistics.mean(x), n = 5)
    Q_calc_autocorr = []
    for i in x:
        i = i ** 2
        Q_calc_autocorr.append(i)
    Q_value = len(x) * sum(Q_calc_autocorr)
    return Q_value

#calling auto-corelation function
np_acf_calc_residuals = auto_correlation(training_residual)
var1 =np.arange(0,20)
m=1.96/np.sqrt(len(data.pollution))
plt.stem(var1,np_acf_calc_residuals,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*var1,np_acf_calc_residuals,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Multiple Linear Regression Model')
plt.axhspan(-m,m,alpha = .1, color = 'green')
plt.tight_layout()
plt.show()
#calling function to calculate Q values
Q  = (len(training_residual)) *(np.sum(np.square(np_acf_calc_residuals)))
print(f"The Q value of residual of regression is {Q}")
print(f"The mean of residuals is {np.mean(training_residual)}")
print(f"The variance of residual is {np.var(training_residual)}")
testing_error_regression= np.subtract(y_test, prediction_test)
MSE = np.square(testing_error_regression).mean()
print("\nMean square error for testing set multiple regression is ", MSE)
print(f"RMSE for testing set using multiple regression is :{np.sqrt(MSE)} ")
print(f"The mean of forecast of multiple regression is {np.mean(testing_error_regression)}")
print(f"The variance of foreacast of multiple regression is {np.var(testing_error_regression)}")
print(f"\n The ratio of resid vs forecast is {np.var(training_residual)/np.var(testing_error_regression)}")

plt.plot(y_test, label = 'Test set')
plt.plot(prediction_test, label = ' One-step prediction using multiple regression method')
plt.xlabel("Time(Daily)")
plt.ylabel("Pollution")
plt.title("Plot of Pollution(pm2.5) prediction using Regression Method")
plt.legend()
plt.show()

#Base Models:

#Average Method:

y_predict_train_set = []
value = 0
for i in range(len(y_train)):
    if i != 0:
        value = value + y_train[i - 1]
        t_value  = i
        y_each_predict = value / i
        y_predict_train_set.append(y_each_predict)
    else:
        continue

y_predict_test_set= []
for i in range(len(y_test)):
    y_predict_each = sum(y_train) / len(y_train)
    y_predict_test_set.append(y_predict_each)

y_preidction_average= pd.DataFrame(y_predict_test_set).set_index(y_test.index)
plt.plot(y_train, label = 'Training set')
plt.plot(y_test, label = 'Test set')
plt.plot(y_preidction_average, label = 'forecast using average method')
plt.xlabel("Time(Daily)")
plt.ylabel("Pollution")
plt.title("Plot of Pollution prediction(pm2.5) using Average Method")
plt.legend()
plt.show()
#now lets find out the MSE of our prediction error --on training set using average method
error_train_set_avg = np.subtract(y_train[1:], y_predict_train_set)
#print(error_train_set_avg)
def calc_MSE(x):
    MSE = np.square(np.array(x)).mean()
    return MSE
MSE_train_set =calc_MSE(error_train_set_avg)
#MSE_train_set = np.sum((var_square_train_set_array) ** 2 )/ len(t_train_set)
print(f"\nMSE of prediction error (training set) using average method is : {MSE_train_set}")
#calling auto-corelation function
np_acf_calc_residuals_average = auto_correlation(error_train_set_avg)
var1 =np.arange(0,20)
m=1.96/np.sqrt(len(data.pollution))
plt.stem(var1,np_acf_calc_residuals_average,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*var1,np_acf_calc_residuals_average,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Average Method')
plt.axhspan(-m,m,alpha = .1, color = 'green')
plt.tight_layout()
plt.show()
#calling function to calculate Q values
Q_residual  = (len(error_train_set_avg)) *(np.sum(np.square(np_acf_calc_residuals_average)))
print(f"The Q value of residual using average method is {Q_residual}")
print(f"The mean of residuals using average method is {np.mean(error_train_set_avg)}")
print(f"The variaince of residual using average method is {np.var(error_train_set_avg)}")
error_test_set_avg = np.subtract(y_test, y_predict_test_set)
MSE_train_set =calc_MSE(error_test_set_avg)
print(f"\nMSE of forecast (testing set) using average method is : {MSE_train_set}")
print(f"Mean of forecast error is: {np.mean(np.array(error_test_set_avg))}")
print(f"Variance of forecast error is: {np.var(np.array(error_test_set_avg))}")
print(f"\n The ratio of resid vs forecast of average method is {( np.var(error_train_set_avg) ) / (  np.var(np.array(error_test_set_avg))  )}")

#Naive method:

print("****  Naive Method   ****")
y_predict_train_set_naive = []
value = 0
for i in range(len(y_train[1:])):
    y_predict_train_set_naive.append(y_train[i])
#print(y_predict_train_set_naive)
y_predict_test_set_naive= [y_train[-1] for i in y_test]
y_prediction_naive_test= pd.DataFrame(y_predict_test_set_naive).set_index(y_test.index)
#print(y_predict_test_set_naive)
plt.plot(y_train, label = 'Training set')
plt.plot(y_test, label = 'Test set')
plt.plot(y_prediction_naive_test, label = 'Forecast using naive method')
plt.xlabel("Time(Daily)")
plt.ylabel("Pollution")
plt.title("Plot of Pollution prediction(pm2.5) using Naive method")
plt.legend()
plt.show()
error_train_set_naive = np.subtract(y_train[1:], y_predict_train_set_naive)
error_test_set_naive = np.subtract(y_test, y_predict_test_set_naive)
MSE_train_set_naive =calc_MSE(error_train_set_naive)
print(f"\nMSE of prediction error (training set) using naive method is : {MSE_train_set_naive}")
#calling auto-corelation function
np_acf_calc_residuals_naive = auto_correlation(error_train_set_naive)
var1 =np.arange(0,20)
m=1.96/np.sqrt(len(data.pollution))
plt.stem(var1,np_acf_calc_residuals_naive,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*var1,np_acf_calc_residuals_naive,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Naive Method')
plt.axhspan(-m,m,alpha = .1, color = 'green')
plt.tight_layout()
plt.show()

#calling function to calculate Q values
Q_residual  = (len(error_train_set_naive)) *(np.sum(np.square(np_acf_calc_residuals_naive)))
print(f"The Q value of residual using naive method is {Q_residual}")
print(f"The mean of residuals using naive method is {np.mean(error_train_set_naive)}")
print(f"The variance of residual using naive method is {np.var(error_train_set_naive)}")
MSE_test_set_naive =calc_MSE(error_test_set_naive)
print(f"\nMSE of prediction error (testing set) using naive method is : {MSE_test_set_naive}")
print(f"Mean of forecast error using naive method is: {np.mean(np.array(error_test_set_naive))}")
print(f"Variance of forecast error using naive method is: {np.var(np.array(error_test_set_naive))}")
print(f"\n The ratio of resid vs forecast of naive method is {( np.var(error_train_set_naive) ) / ( np.var(np.array(error_test_set_naive)) )}")

#Drift method

print("***** Drift Method   ********")
y_predict_train_set_drift = []
value = 0
for i in range(len(y_train)):
    if i > 1:
        slope_val = (y_train[i - 1] - y_train[0]) / (i-1)
        y_each_predict = (slope_val * i) + y_train[0]
        y_predict_train_set_drift.append(y_each_predict)
    else:
        continue
y_predict_test_set_drift= []
for h in range(len(y_test)):
    slope_val = (y_train[-1] - y_train[0] ) /( len(y_train) - 1 )
    y_predict_each = y_train[-1] + ((h +1) * slope_val)
    y_predict_test_set_drift.append(y_predict_each)

y_preidction_drift= pd.DataFrame(y_predict_test_set_drift).set_index(y_test.index)
plt.plot(y_train, label = 'Training set')
plt.plot(y_test, label = 'Test set')
plt.plot(y_preidction_drift, label = 'forecast using drift method')
plt.xlabel("Time(Daily)")
plt.ylabel("Pollution")
plt.title("Plot of Pollution prediction(pm2.5) using Drift method")
plt.legend()
plt.show()
error_train_set_drift = np.subtract(y_train[2:], y_predict_train_set_drift)
error_test_set_drift = np.subtract(y_test, y_predict_test_set_drift)
MSE_train_set_drift =calc_MSE(error_train_set_drift)
print(f"\nMSE of prediction error (training set) using drift method is : {MSE_train_set_drift}")
#calling auto-corelation function
np_acf_calc_residuals_drift = auto_correlation(error_train_set_drift)
var1 =np.arange(0,20)
m=1.96/np.sqrt(len(data.pollution))
plt.stem(var1,np_acf_calc_residuals_drift,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*var1,np_acf_calc_residuals_drift,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Drift Method')
plt.axhspan(-m,m,alpha = .1, color = 'green')
plt.tight_layout()
plt.show()
#calling function to calculate Q values
Q_residual  = (len(error_train_set_drift)) *(np.sum(np.square(np_acf_calc_residuals_drift)))
print(f"The Q value of residual using drift method is {Q_residual}")
print(f"The mean of residuals using drift method is {np.mean(error_train_set_drift)}")
print(f"The variance of residual using drift method is {np.var(error_train_set_drift)}")
MSE_test_set_drift =calc_MSE(error_test_set_drift)
print(f"\nMSE of prediction error of testing set using drift method is : {MSE_test_set_drift}")
print(f"Mean of forecast error using drift method is: {np.mean(np.array(error_test_set_drift))}")
print(f"Variance of forecast error using drift method is: {np.var(np.array(error_test_set_drift))}")
print(f"\n The ratio of resid vs forecast of drift method is {( np.var(error_train_set_drift) ) / ( np.var(np.array(error_test_set_drift)) )}")

#Simple and exponential smoothing
SES = ets.ExponentialSmoothing(y_train,trend=None,damped=False,seasonal=None).fit()
SES_predict_train= SES.forecast(steps=len(y_train))
SES_predict_test= SES.forecast(steps=len(y_test))
predict_test_SES = pd.DataFrame(SES_predict_test).set_index(y_test.index)
resid_SES = np.subtract(y_train.values,np.array(SES_predict_train))
forecast_error_Ses = np.subtract(y_test.values,np.array(SES_predict_test))
MSE_SES = np.square(resid_SES).mean()
print("Mean square error for (training set) simple exponential smoothing is ", MSE_SES)
plt.plot(y_train,label= "Air Qaulity-train")
plt.plot(y_test,label= "Air Quality-test")
plt.plot(predict_test_SES,label= "SES Method prediction")
plt.legend(loc='upper left')
plt.title('SES method for Pollution(pm2.5) prediction')
plt.xlabel('Time(Daily)')
plt.ylabel('Pollution')
plt.show()
#calling auto-corelation function
np_acf_calc_residuals_SES = auto_correlation(resid_SES)
var1 =np.arange(0,20)
m=1.96/np.sqrt(len(data.pollution))
plt.stem(var1,np_acf_calc_residuals_SES,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*var1,np_acf_calc_residuals_SES,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of SES Method')
plt.axhspan(-m,m,alpha = .1, color = 'green')
plt.tight_layout()
plt.show()
#calling function to calculate Q values
Q_residual  = (len(resid_SES)) *(np.sum(np.square(np_acf_calc_residuals_SES)))
print(f"The Q value of residual using SES method is {Q_residual}")
print(f"Mean of residual using SES method is: {np.mean(np.array(resid_SES))}")
print(f"Variance of residual using SES method is: {np.var(np.array(resid_SES))}")
MSE_SES = np.square(forecast_error_Ses).mean()
print("Mean square error for (testing set) simple exponential smoothing is ", MSE_SES)
print(f"Mean of forecast error using SES method is: {np.mean(np.array(forecast_error_Ses))}")
print(f"Variance of forecast error using SES method is: {np.var(np.array(forecast_error_Ses))}")
print(f"\n The ratio of resid vs forecast of SES method is {( np.var(np.array(resid_SES)) ) / ( np.var(np.array(forecast_error_Ses)) )}")


#ARMA and ARIMA and SARIMA model:
#a. Preliminary model development procedures and results. (ARMA model order
#determination). Pick at least two orders using GPAC table.
#b. Should include discussion of the autocorrelation function and the GPAC. Include a plot of
#the autocorrelation function and the GPAC table within this section).
#c. Include the GPAC table in your report and highlight the estimated order.
ry = auto_correlation(data_diff1)
ry1 = ry[::-1]
ry2 = np.concatenate((np.reshape(ry1,20),ry[1:]))
na_order =8
nb_order =8
calc_Gpac(na_order, nb_order, ry2)

#Since we dont see any patterns after feeding the differenced variable to the auto-correlation which proves that we dont have ARIMA model so we will be feeding the dependant variable to check the gpac pattern.
y = data['pollution']
ry = auto_correlation(y)
ry1 = ry[::-1]
ry2 = np.concatenate((np.reshape(ry1,20),ry[1:]))
na_order =8
nb_order =8
calc_Gpac(na_order, nb_order, ry2)

#we will perform ARMA(2,0) looking at the pattern obtained from the gpac.

delta = 10**-6
na =2
nb =0
n = na +nb
theta = np.zeros(n)
u = 0.01
u_max = 1e10
count = 60

print("\nFor our estimated ARMA (2,O): ")

theta, cov,SSE_count = calc_LMA(count, dependant_variable, na,nb, theta, delta, u, u_max)
Y_train , Y_test = train_test_split(y, test_size= 0.2, shuffle =False)
y_predict=[]
for i in range(len(Y_train)):
    if i ==0:
        predict = (-theta[0]) * Y_train[i]
    else:
        predict = ( - theta[0] * Y_train[i] ) + ( -theta[1] * Y_train[i-1])
    y_predict.append(predict)

resid = np.subtract(np.array(Y_train),np.array(y_predict))
acf_resid_arma = auto_correlation(resid)
var1 =np.arange(0,20)
m=1.96/np.sqrt(len(data.pollution))
plt.stem(var1,acf_resid_arma,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*var1,acf_resid_arma,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of ARMA(2,0)')
plt.axhspan(-m,m,alpha = .1, color = 'green')
plt.tight_layout()
plt.show()
Q  = (len(resid)) *(np.sum(np.square(acf_resid_arma)))
DOF = len(resid) - na -nb
alfa =0.01
chi_critical = chi2.ppf(1-alfa, DOF)
print("\nThe Q value is  ",Q)
print(f"The chi-critical is {chi_critical}")

if Q <chi_critical:
    print(f"\n The Q value is less than {chi_critical} (chi-critical) so, the residual is white")
else:
    print(f"\nThe Q value is not less than {chi_critical} (chi-critical) so, the residual is not white")

print(resid[:5])
print(f"\nMSE of training data for ARMA(2,0) is {np.square(resid).mean()}")
print(f"Mean of residual(training data) with ARMA(2,0) is {np.mean(resid)}")
print(f"The variance of residual(training data) with ARMA(2,0) is {np.var(resid)}")
y_predict = pd.DataFrame(y_predict).set_index(Y_train.index)
plt.plot(Y_train, label='Y_train')
plt.plot(y_predict, label ='Predicted values')
plt.xlabel('Number of observations')
plt.ylabel('y-values')
plt.title("One-step-ahead prediction for ARMA(2,0)")
plt.legend()
plt.show()

# prediction for test set
y_prediction_test=[]
for i in range(len(Y_test)):
    if i ==0:
        predict = (-theta[0] * Y_train[-1])+ (-theta[1] * Y_train[-2])
    elif i ==1:
        predict = ( -theta[0] * y_prediction_test[0] ) + (-theta[1] * Y_train[-1])
    elif i ==2:
        predict = ( -theta[0] * y_prediction_test[1] ) + (-theta[1] * y_prediction_test[0])
    else:
        predict = (-theta[0] * y_prediction_test[i - 1]) + (-theta[1] * y_prediction_test[i-2])
    y_prediction_test.append(predict)
y_prediction_test = pd.DataFrame(y_prediction_test).set_index(Y_test.index)
forecast_error = np.subtract(np.array(Y_test), np.array(y_prediction_test))
print(f"\nMSE of forecast for ARMA(2,0) is {np.square(forecast_error).mean()}")
print(f"The mean of testing data is {np.mean(forecast_error)}")
print(f"The variance of testing data is {np.var(forecast_error)}")
ratio = np.var(resid)/np.var(forecast_error)
print(f"\nThe ratio of variance of residual to variance of forecast is {ratio}")
plt.plot(Y_train, label='Training set')
plt.plot(Y_test, label = 'Testing set')
plt.plot(y_prediction_test, label ='Prediction Test set')
plt.xlabel("Time(Daily)")
plt.ylabel("Pollution")
plt.title("Plot of Pollution(pm2.5) prediction using ARMA(2,0) ")
plt.legend()
plt.show()

#Second pattern of gpac(2,1):

delta = 10**-6
na =2
nb =1
n = na +nb
theta = np.zeros(n)
u = 0.01
u_max = 1e10
count = 60

print("\nFor our estimated ARMA (2,1): ")
theta, cov,SSE_count = calc_LMA(count, dependant_variable, na,nb, theta, delta, u, u_max)
Y_train , Y_test = train_test_split(y, test_size= 0.2, shuffle =False)
y_predict=[]
for i in range(len(Y_train)):
    if i ==0:
        predict = (-theta[0]) * Y_train[i]
    else:
        predict = ( - theta[0] * Y_train[i] ) + ( -theta[1] * Y_train[i-1])
    y_predict.append(predict)

resid = np.subtract(np.array(Y_train),np.array(y_predict))
acf_resid_ar = auto_correlation(resid)
var1 =np.arange(0,20)
m=1.96/np.sqrt(len(data.pollution))
plt.stem(var1,acf_resid_ar,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*var1,acf_resid_ar,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of ARMA(2,1)')
plt.axhspan(-m,m,alpha = .1, color = 'green')
plt.tight_layout()
plt.show()
Q  = (len(resid)) *(np.sum(np.square(acf_resid_ar)))
DOF = len(resid) - na -nb
alfa =0.01
chi_critical = chi2.ppf(1-alfa, DOF)
print("\nThe Q value is  ",Q)
print(f"The chi-critical is {chi_critical}")

if Q <chi_critical:
    print(f"\n The Q value is less than {chi_critical} (chi-critical) so, the residual is white")
else:
    print(f"\nThe Q value is not less than {chi_critical} (chi-critical) so, the residual is not white")

print(resid[:5])
print(f"\nMSE of training data for ARMA(2,1) is {np.square(resid).mean()}")
print(f"Mean of residual(training data) with ARMA(2,1) is {np.mean(resid)}")
print(f"The variance of residual(training data) with ARMA(2,1) is {np.var(resid)}")
y_predict = pd.DataFrame(y_predict).set_index(Y_train.index)
plt.plot(Y_train, label='Y_train')
plt.plot(y_predict, label ='Predicted values')
plt.xlabel('Number of observations')
plt.ylabel('y-values')
plt.title("One-step-ahead prediction for ARMA(2,1)")
plt.legend()
plt.show()

# prediction for test set
y_prediction_test=[]
for i in range(len(Y_test)):
    if i ==0:
        predict = (-theta[0] * Y_train[-1])+ (-theta[1] * Y_train[-2])
    elif i ==1:
        predict = ( -theta[0] * y_prediction_test[0] ) + (-theta[1] * Y_train[-1])
    elif i ==2:
        predict = ( -theta[0] * y_prediction_test[1] ) + (-theta[1] * y_prediction_test[0])
    else:
        predict = (-theta[0] * y_prediction_test[i - 1]) + (-theta[1] * y_prediction_test[i-2])
    y_prediction_test.append(predict)

y_prediction_test = pd.DataFrame(y_prediction_test).set_index(Y_test.index)
forecast_error = np.subtract(np.array(Y_test), np.array(y_prediction_test))
print(f"\nMSE of forecast for ARMA(2,1) is {np.square(forecast_error).mean()}")
print(f"The mean of testing set error is {np.mean(forecast_error)}")
print(f"The variance of testing set error is {np.var(forecast_error)}")
ratio = np.var(resid)/np.var(forecast_error)
print(f"\nThe ratio of variance of residual to variance of forecast is {ratio}")
plt.plot(Y_train, label='Training set')
plt.plot(Y_test, label = 'Testing set')
plt.plot(y_prediction_test, label ='Prediction-Test-set')
plt.xlabel("Time(Daily)")
plt.ylabel("Pollution")
plt.title("Plot of Pollution(pm2.5) prediction using ARMA(2,1) ")
plt.legend()
plt.show()

#Forecast function:

print("****  Drift Method   ****")

def forecast_function(Y_train, step):
    y_predict_test_set_drift= []
    for h in range(step):
        slope_val = (Y_train[-1] - Y_train[0] ) /( len(Y_train) - 1 )
        y_predict_each = Y_train[-1] + ((h +1) * slope_val)
        y_predict_test_set_drift.append(y_predict_each)

    return y_predict_test_set_drift

step = len(Y_test)
y_predict_test_set_drift = forecast_function(Y_train, step)
y_predict_test_set_drift = pd.DataFrame(y_predict_test_set_drift).set_index(Y_test.index)
plt.plot(Y_train, label = 'Training set')
plt.plot(Y_test, label = 'Test set')
plt.plot(y_predict_test_set_drift, label = ' Forecast using drift method')
plt.xlabel("Time(Daily)")
plt.ylabel("Pollution(pm2.5-concentration)")
plt.title("Plot of Pollution prediction using drift method")
plt.legend()
plt.show()

#============ H-step prediction =======================================

step =100
y_predict_h_step_drift = forecast_function(Y_train, step)
y_predict_h_step_drift= pd.DataFrame(y_predict_h_step_drift).set_index(Y_test[:100].index)
plt.plot(Y_test[:100], label = 'Test set')
plt.plot(y_predict_h_step_drift, label = ' 100 -step forecast using drift method')
plt.xlabel("Time(daily)")
plt.ylabel("Pollution(pm2.5-concentration)")
plt.title("Plot of H-step prediction")
plt.legend()
plt.show()



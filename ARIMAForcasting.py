import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import pandas as pd, numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

columns = ['Index', 'MEASURED_DEPTH', 'STEP', 'TIME_DRILLING', 'VERTICAL_DEPTH', 'TOT_GAS', 'C1_OUT', 'C2_OUT', 'C3_OUT', 'iC4_OUT', 'C4_TOT_OUT', 'nC4_OUT', 'ROP', 'WOB', 'RPM', 'iC5_OUT', 'C5_TOT_OUT','nC5_OUT','CVD_FLOW','GDS_FLOW','TORQUE_MAX','TORQUE','TORQUE_MIN','MW_IN','SPP','ETHYLENE','PROPYLENE']
raw_data = pd.read_csv("well_1_a.csv", index_col="Index")
df = pd.DataFrame(raw_data, columns=['Index', 'MEASURED_DEPTH', 'STEP', 'TIME_DRILLING', 'VERTICAL_DEPTH', 'TOT_GAS', 'C1_OUT', 'C2_OUT', 'C3_OUT', 'iC4_OUT', 'C4_TOT_OUT', 'nC4_OUT', 'ROP', 'WOB', 'RPM', 'iC5_OUT', 'C5_TOT_OUT','nC5_OUT','CVD_FLOW','GDS_FLOW','TORQUE_MAX','TORQUE','TORQUE_MIN','MW_IN','SPP','ETHYLENE','PROPYLENE']
)
df = df.drop(df.index[0])
df = df[columns].apply(pd.to_numeric)
df[df < 0] = np.nan

df['ROP'] = df['ROP'].replace(to_replace=np.nan, value=np.mean(df['ROP']), method='pad')
df['ROP'] = df['ROP'].replace(to_replace=0, value=np.mean(df['ROP']), method='pad')
df['RPM'] = df['RPM'].replace(to_replace=np.nan, value=np.mean(df['RPM']), method='pad')
df['TOT_GAS'] = df['TOT_GAS'].replace(to_replace=np.nan, value=np.mean(df['TOT_GAS']), method='pad')
df['SPP'] = df['SPP'].replace(to_replace=np.nan, value=np.mean(df['SPP']), method='pad')
df['WOB'] = df['WOB'].replace(to_replace=np.nan, value=np.mean(df['WOB']), method='pad')
df['TORQUE'] = df['TORQUE'].replace(to_replace=np.nan, value=np.mean(df['TORQUE']), method='pad')
df['VERTICAL_DEPTH'] = df['VERTICAL_DEPTH'].replace(to_replace=np.nan, value=np.mean(df['VERTICAL_DEPTH']), method='pad')
df['MEASURED_DEPTH'] = df['MEASURED_DEPTH'].replace(to_replace=np.nan, value=np.mean(df['MEASURED_DEPTH']), method='pad')
df['C1_OUT'] = df['C1_OUT'].replace(to_replace=np.nan, value=np.mean(df['C1_OUT']), method='pad')
df['C2_OUT'] = df['C2_OUT'].replace(to_replace=np.nan, value=np.mean(df['C2_OUT']), method='pad')
df['C3_OUT'] = df['C3_OUT'].replace(to_replace=np.nan, value=np.mean(df['C3_OUT']), method='pad')
df['iC4_OUT'] = df['iC4_OUT'].replace(to_replace=np.nan, value=np.mean(df['iC4_OUT']), method='pad')
df['iC5_OUT'] = df['iC5_OUT'].replace(to_replace=np.nan, value=np.mean(df['iC5_OUT']), method='pad')
df['nC5_OUT'] = df['nC5_OUT'].replace(to_replace=np.nan, value=np.mean(df['nC5_OUT']), method='pad')
df['nC4_OUT'] = df['nC4_OUT'].replace(to_replace=np.nan, value=np.mean(df['nC4_OUT']), method='pad')
df['C4_TOT_OUT'] = df['C4_TOT_OUT'].replace(to_replace=np.nan, value=np.mean(df['C4_TOT_OUT']), method='pad')
df['C5_TOT_OUT'] = df['C5_TOT_OUT'].replace(to_replace=np.nan, value=np.mean(df['C5_TOT_OUT']), method='pad')

plt.plot(df['ROP'], df['ROP'].values, color='blue', label='Original-data')
plt.show()



def check_stationarity(df):
    # Determing rolling statistics
    rolling_mean = df.rolling(window=52, center=False).mean()
    rolling_std = df.rolling(window=52, center=False).std()

    # Plot rolling statistics:
    plt.plot(df, df.values, color='blue', label='Original')
    plt.plot(rolling_mean, rolling_mean.values, color='red', label='Rolling Mean')
    plt.plot(rolling_std, rolling_std.values, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dickey_fuller_test = adfuller(df, autolag='AIC')
    dfresults = pd.Series(dickey_fuller_test[0:4],
                          index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dickey_fuller_test[4].items():
        dfresults['Critical Value (%s)' % key] = value
    print (dfresults)

res = check_stationarity(df['ROP'])
print(res)



#Make data stationary
dfl = np.log(df['ROP'])
plt.plot(dfl, dfl.values, color='blue', label='stationary-data')
plt.show()

'''
moving_avg = df.rolling(window=12, center=False).mean()
ts_log_moving_avg_diff = dfl - moving_avg
print ts_log_moving_avg_diff.head(12)
ts_log_moving_avg_diff.dropna(inplace=True)
res2 = check_stationarity(ts_log_moving_avg_diff)
print(res2) 
'''

expwighted_avg = pd.ewma(dfl, halflife=12)
plt.plot(expwighted_avg, expwighted_avg.values ,color='red', label='expwighted')
plt.show()

ts_log_ewma_diff = dfl - expwighted_avg
print(ts_log_ewma_diff.head(12))
res3= check_stationarity(ts_log_ewma_diff)
print(res3)



#Differencing
ts_log_diff = dfl - dfl.shift()
ts_log_diff.dropna(inplace=True)
plt.plot(ts_log_diff, ts_log_diff.values, color='orange', label='log-diff')
plt.show()



#Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(dfl, freq=3)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(dfl, dfl.values, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, trend.values, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, seasonal.values,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, residual.values, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
res4 = check_stationarity(ts_log_decompose)
print(res4)



#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()




# ARIMA model
model = ARIMA(dfl, order=(0,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
plt.plot(ts_log_diff, ts_log_diff.values, color='green', label='log-diff')
#plt.plot(model_fit, model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((model_fit.fittedvalues-ts_log_diff)**2))
plt.show()

predictions_ARIMA_diff = pd.Series(model_fit.fittedvalues, copy=True)
print predictions_ARIMA_diff.head()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print predictions_ARIMA_diff_cumsum.head()

predictions_ARIMA_log = pd.Series(dfl.ix[0], index=dfl.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
print predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(dfl, dfl.values, color='orange', label='original')
plt.plot(predictions_ARIMA, predictions_ARIMA.values, color='blue')
plt.show()

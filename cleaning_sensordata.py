import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import stats

columns = ['Index', 'MEASURED_DEPTH', 'STEP', 'TIME_DRILLING', 'VERTICAL_DEPTH', 'TOT_GAS', 'C1_OUT', 'C2_OUT', 'C3_OUT', 'iC4_OUT', 'C4_TOT_OUT', 'nC4_OUT', 'ROP', 'WOB', 'RPM', 'iC5_OUT', 'C5_TOT_OUT','nC5_OUT','CVD_FLOW','GDS_FLOW','TORQUE_MAX','TORQUE','TORQUE_MIN','MW_IN','SPP','ETHYLENE','PROPYLENE']
raw_data = pd.read_csv("well_1_a.csv", index_col="Index")
df = pd.DataFrame(raw_data, columns = ['Index', 'MEASURED_DEPTH', 'STEP', 'TIME_DRILLING', 'VERTICAL_DEPTH', 'TOT_GAS', 'C1_OUT', 'C2_OUT', 'C3_OUT', 'iC4_OUT', 'C4_TOT_OUT', 'nC4_OUT', 'ROP', 'WOB', 'RPM', 'iC5_OUT', 'C5_TOT_OUT','nC5_OUT','CVD_FLOW','GDS_FLOW','TORQUE_MAX','TORQUE','TORQUE_MIN','MW_IN','SPP','ETHYLENE','PROPYLENE']
)
df = df.drop(df.index[0])
df = df[columns].apply(pd.to_numeric)
#df[df != -999.25] = 0
#df = df.replace(to_replace=-999.25, value=1, method='pad')

#df.plot(y=["TIME_DRILLING"], sharey=True, kind='line')
#plt.savefig('TIME_DRILLING-NAN.png')
#df.plot(y=["STEP"], sharey=True, kind='line')
#plt.savefig('STEP-NAN.png')
#df.plot(y=["VERTICAL_DEPTH"], sharey=True, kind='line')
#plt.savefig('VERTICAL_DEPTH-NAN.png')
#df.plot(y=["TOT_GAS"], sharey=True, kind='line')
#plt.savefig('TOT_GAS-NAN.png')
#df.plot(y=["C1_OUT"], sharey=True, kind='line')
#plt.savefig('C1_OUT-NAN.png')
#df.plot(y=["C2_OUT"], sharey=True, kind='line')
#plt.savefig('C2_OUT-NAN.png')
#df.plot(y=["C3_OUT"], sharey=True, kind='line')
#plt.savefig('C3_OUT-NAN.png')
#df.plot(y=["iC4_OUT"], sharey=True, kind='line')
#plt.savefig('iC4_OUT-NAN.png')
#df.plot(y=["C4_TOT_OUT"], sharey=True, kind='line')
#plt.savefig('C4_TOT_OUT-NAN.png')
#df.plot(y=["nC4_OUT"], sharey=True, kind='line')
#plt.savefig('nC4_OUT-NAN.png')
#df.plot(y=["iC5_OUT"], sharey=True, kind='line')
#plt.savefig('iC5_OUT-NAN.png')
#df.plot(y=["C5_TOT_OUT"], sharey=True, kind='line')
#plt.savefig('C5_TOT_OUT-NAN.png')
#df.plot(y=["nC5_OUT"], sharey=True, kind='line')
#plt.savefig('nC5_OUT-NAN.png')
#df.plot(y=["CVD_FLOW"], sharey=True, kind='line')
#plt.savefig('CVD_FLOW-NAN.png')
#df.plot(y=["GDS_FLOW"], sharey=True, kind='line')
#plt.savefig('GDS_FLOW-NAN.png')
#df.plot(y=["TORQUE_MAX"], sharey=True, kind='line')
#plt.savefig('TORQUE_MAX-NAN.png')
#df.plot(y=["TORQUE"], sharey=True, kind='line')
#plt.savefig('TORQUE-NAN.png')
#df.plot(y=["TORQUE_MIN"], sharey=True, kind='line')
#plt.savefig('TORQUE_MIN-NAN.png')
#df.plot(y=["MW_IN"], sharey=True, kind='line')
#plt.savefig('MW_IN-NAN.png')
#df.plot(y=["SPP"], sharey=True, kind='line')
#plt.savefig('SPP-NAN.png')
#df.plot(y=["ETHYLENE"], sharey=True, kind='line')
#plt.savefig('ETHYLENE-NAN.png')
#df.plot(y=["PROPYLENE"], sharey=True, kind='line')
#plt.savefig('PROPYLENE-NAN.png')
#df.plot(y=["ROP"], sharey=True, kind='line')
#plt.savefig('ROP-NAN.png')
#df.plot(y=["RPM"], sharey=True, kind='line')
#plt.savefig('RPM-NAN.png')
#df.plot(y=["WOB"], sharey=True, kind='line')
#plt.savefig('WOB-NAN.png')
#plt.show()

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




#print(stats.zscore(df['TOT_GAS']))
#df.plot(y=["TIME_DRILLING"], sharey=True, kind='line')
#plt.savefig('TIME_DRILLING-orginal.png')
#df.plot(y=["STEP"], sharey=True, kind='line')
#plt.savefig('STEP-original.png')
#df.plot(y=["VERTICAL_DEPTH"], sharey=True, kind='line')
#plt.savefig('VERTICAL_DEPTH-original.png')
#df.plot(y=["TOT_GAS"], sharey=True, kind='line')
#plt.savefig('TOT_GAS-original.png')
#df.plot(y=["C1_OUT"], sharey=True, kind='line')
#plt.savefig('C1_OUT-original.png')
#df.plot(y=["C2_OUT"], sharey=True, kind='line')
#plt.savefig('C2_OUT-original.png')
#df.plot(y=["C3_OUT"], sharey=True, kind='line')
#plt.savefig('C3_OUT-original.png')
#df.plot(y=["iC4_OUT"], sharey=True, kind='line')
#plt.savefig('iC4_OUT-original.png')
#df.plot(y=["C4_TOT_OUT"], sharey=True, kind='line')
#plt.savefig('C4_TOT_OUT-original.png')
#df.plot(y=["nC4_OUT"], sharey=True, kind='line')
#plt.savefig('nC4_OUT-original.png')
#df.plot(y=["iC5_OUT"], sharey=True, kind='line')
#plt.savefig('iC5_OUT-original.png')
#df.plot(y=["C5_TOT_OUT"], sharey=True, kind='line')
#plt.savefig('C5_TOT_OUT-original.png')
#df.plot(y=["nC5_OUT"], sharey=True, kind='line')
#plt.savefig('nC5_OUT-original.png')
#df.plot(y=["CVD_FLOW"], sharey=True, kind='line')
#plt.savefig('CVD_FLOW-original.png')
#df.plot(y=["GDS_FLOW"], sharey=True, kind='line')
#plt.savefig('GDS_FLOW-original.png')
#df.plot(y=["TORQUE_MAX"], sharey=True, kind='line')
#plt.savefig('TORQUE_MAX-original.png')
#df.plot(y=["TORQUE"], sharey=True, kind='line')
#plt.savefig('TORQUE-original.png')
#df.plot(y=["TORQUE_MIN"], sharey=True, kind='line')
#plt.savefig('TORQUE_MIN-original.png')
#df.plot(y=["MW_IN"], sharey=True, kind='line')
#plt.savefig('MW_IN-original.png')
#df.plot(y=["SPP"], sharey=True, kind='line')
#plt.savefig('SPP-original.png')
#df.plot(y=["ETHYLENE"], sharey=True, kind='line')
#plt.savefig('ETHYLENE-original.png')
#df.plot(y=["PROPYLENE"], sharey=True, kind='line')
#plt.savefig('PROPYLENE-original.png')
#df.plot(y=["ROP"], sharey=True, kind='line')
#plt.savefig('ROP-original.png')
#df.plot(y=["RPM"], sharey=True, kind='line')
#plt.savefig('RPM-original.png')
#df.plot(y=["WOB"], sharey=True, kind='line')
#plt.savefig('WOB-original.png')
plt.show()


#df['TIME_DRILLING'] = df['TIME_DRILLING'].interpolate()
#df.plot(y=["TIME_DRILLING"], sharey=True, kind='line')
#plt.savefig('TIME_DRILLING-interpolated_a.png')
#plt.show()
#print(len(df[df['TIME_DRILLING'] > 0.6]))
#print(df[df['TIME_DRILLING'] > 0.6].head())
#df['TIME_DRILLING'] = np.array(df['TIME_DRILLING'].tolist())
#df['TIME_DRILLING'] = np.where(df['TIME_DRILLING'] > 0.6, 0.6 , df['TIME_DRILLING']).tolist()
#df.plot(y=["TIME_DRILLING"], sharey=True, kind='line')
#plt.savefig('TIME_DRILLING-interpolatedthr_a.png')
#print('mean_timedrilling:',df['TIME_DRILLING'].mean())
#print('min_timedrilling:',df['TIME_DRILLING'].min())
#print('max_timedrilling:',df['TIME_DRILLING'].max())
#print('std_timedrilling:',df['TIME_DRILLING'].std())

#df['TOT_GAS'] = df['TOT_GAS'].interpolate(method='polynomial', order=2)
#df.plot(y=["TOT_GAS"], sharey=True, kind='line')
#plt.savefig('TOT_GAS-interpolated1_a.png')
#plt.show()
#print(len(df[df['TOT_GAS'] > 100000]))
#print(df[df['TOT_GAS'] > 100000].head())
#df['TOT_GAS'] = np.array(df['TOT_GAS'].tolist())
#df['TOT_GAS'] = np.where(df['TOT_GAS'] > 100000, 100000 , df['TOT_GAS']).tolist()
#df.plot(y=["TOT_GAS"], sharey=True, kind='line')
#plt.savefig('TOT_GAS-interpolated_a.png')
#print('mean_totgas:',df['TOT_GAS'].mean())
#print('min_totgas:',df['TOT_GAS'].min())
#print('max_totgas:',df['TOT_GAS'].max())
#print('std_totgas:',df['TOT_GAS'].std())
#print(df['TOT_GAS'])
'''
col = df['ROP']
def general_trend(col):
    median = np.median(col)
    return median

def u_t(col):
    median = np.median(col)
    std = np.std(col)
    utr = median + 2*std
    return utr

def l_t(col):
    median = np.median(col)
    std = np.std(col)
    lthr = median - 2*std
    return lthr
'''
#df['TOT_GAS'].rolling(window= 3).mean().plot()
#df['TOT_GAS'].rolling(window= 3).median().plot(color='red')
#df['TOT_GAS'].rolling(window= 3).apply(u_t).plot(color='cyan')
#df['TOT_GAS'].rolling(window= 3).apply(l_t).plot(color='green')
#plt.savefig("TOT_GAS_avg-1a.png")
#df['TOT_GAS'].plot()
#plt.show()
#print(df['TOT_GAS'].mean())
#a = df['TOT_GAS'][df['TOT_GAS'] > np.mean(df['TOT_GAS'])]
#print(a)
#th =df['TOT_GAS'].rolling(window= 3).mean()
#print(th)

#print(df['TOT_GAS'].rolling(window= 3).mean() > np.mean(df['TOT_GAS']))
#df['TOT_GAS'] = df['TOT_GAS'].replace(to_replace=df['TOT_GAS'] > th , value=df['TOT_GAS'].mean(), method='pad')
#print(df['TOT_GAS'])


#df['VERTICAL_DEPTH'] = df['VERTICAL_DEPTH'].interpolate()
#df.plot(y=["VERTICAL_DEPTH"], sharey=True, kind='line')
#plt.savefig('VERTICAL_DEPTH-interpolated1_a.png')
#plt.show()
#df['VERTICAL_DEPTH'].rolling(window= 3).mean().plot()
#df['VERTICAL_DEPTH'].rolling(window= 3).median().plot(color='red')
#plt.savefig("vertical_depth-median-1a.png")
#plt.show()
#print('mean_VERTICAL_DEPTH:',df['VERTICAL_DEPTH'].mean())
#print('min_VERTICAL_DEPTH:',df['VERTICAL_DEPTH'].min())
#print('max_VERTICAL_DEPTH:',df['VERTICAL_DEPTH'].max())
#print('std_VERTICAL_DEPTH:',df['VERTICAL_DEPTH'].std())


#df['MEASURED_DEPTH'] = df['MEASURED_DEPTH'].interpolate()
#df.plot(y=["MEASURED_DEPTH"], sharey=True, kind='line')
#plt.savefig('VERTICAL_DEPTH-interpolated1_a.png')
#plt.show()
#df['VERTICAL_DEPTH'].rolling(window= 3).mean().plot()
#df['MEASURED_DEPTH'].rolling(window= 3).median().plot(color='red')
#plt.savefig("MEASURED_DEPTH-mean-1a.png")
#plt.show()

#df['C1_OUT'] = df['C1_OUT'].interpolate()
#df.plot(y=["C1_OUT"], sharey=True, kind='line')
#plt.savefig('C1_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['C1_OUT'] > 20000]))
#print(df[df['C1_OUT'] > 20000].head())
#df['C1_OUT'] = np.array(df['C1_OUT'].tolist())
#df['C1_OUT'] = np.where(df['C1_OUT'] > 20000, 20000 , df['C1_OUT']).tolist()
#df.plot(y=["C1_OUT"], sharey=True, kind='line')
#plt.savefig('C1_OUT-interpolated_a.png')
#print('mean_C1_OUT:',df['C1_OUT'].mean())
#print('min_C1_OUT:',df['C1_OUT'].min())
#print('max_C1_OUT:',df['C1_OUT'].max())
#print('std_C1_OUT:',df['C1_OUT'].std())


#df['C2_OUT'] = df['C2_OUT'].interpolate()
#df.plot(y=["C2_OUT"], sharey=True, kind='line')
#plt.savefig('C2_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['C2_OUT'] > 15000]))
#print(df[df['C2_OUT'] > 15000].head())
#df['C2_OUT'] = np.array(df['C2_OUT'].tolist())
#df['C2_OUT'] = np.where(df['C2_OUT'] > 15000, 15000 , df['C2_OUT']).tolist()
#df.plot(y=["C2_OUT"], sharey=True, kind='line')
#plt.savefig('C2_OUT-interpolated_a.png')
#print('mean_C2_OUT:',df['C2_OUT'].mean())
#print('min_C2_OUT:',df['C2_OUT'].min())
#print('max_C2_OUT:',df['C2_OUT'].max())
#print('std_C2_OUT:',df['C2_OUT'].std())


#df['C3_OUT'] = df['C3_OUT'].interpolate()
#df.plot(y=["C3_OUT"], sharey=True, kind='line')
#plt.savefig('C3_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['C3_OUT'] > 20000]))
#print(df[df['C3_OUT'] > 20000].head())
#df['C3_OUT'] = np.array(df['C3_OUT'].tolist())
#df['C3_OUT'] = np.where(df['C3_OUT'] > 20000, 20000 , df['C3_OUT']).tolist()
#df.plot(y=["C3_OUT"], sharey=True, kind='line')
#plt.savefig('C3_OUT-interpolated_a.png')
#print('mean_C3_OUT:',df['C3_OUT'].mean())
#print('min_C3_OUT:',df['C3_OUT'].min())
#print('max_C3_OUT:',df['C3_OUT'].max())
#print('std_C3_OUT:',df['C3_OUT'].std())


#df['iC4_OUT'] = df['iC4_OUT'].interpolate()
#df.plot(y=["iC4_OUT"], sharey=True, kind='line')
#plt.savefig('iC4_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['iC4_OUT'] > 2000]))
#print(df[df['iC4_OUT'] > 2000].head())
#df['iC4_OUT'] = np.array(df['iC4_OUT'].tolist())
#df['iC4_OUT'] = np.where(df['iC4_OUT'] > 2000, 2000 , df['iC4_OUT']).tolist()
#df.plot(y=["iC4_OUT"], sharey=True, kind='line')
#plt.savefig('iC4_OUT-interpolated_a.png')
#print('mean_iC4_OUT:',df['iC4_OUT'].mean())
#print('min_iC4_OUT:',df['iC4_OUT'].min())
#print('max_iC4_OUT:',df['iC4_OUT'].max())
#print('std_iC4_OUT:',df['iC4_OUT'].std())



#df['C4_TOT_OUT'] = df['C4_TOT_OUT'].interpolate()
#df.plot(y=["C4_TOT_OUT"], sharey=True, kind='line')
#plt.savefig('C4_TOT_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['C4_TOT_OUT'] > 10000]))
#print(df[df['C4_TOT_OUT'] > 10000].head())
#df['C4_TOT_OUT'] = np.array(df['C4_TOT_OUT'].tolist())
#df['C4_TOT_OUT'] = np.where(df['C4_TOT_OUT'] > 10000, 10000 , df['C4_TOT_OUT']).tolist()
#df.plot(y=["C4_TOT_OUT"], sharey=True, kind='line')
#plt.savefig('C4_TOT_OUT-interpolated_a.png')
#print('mean_C4_TOT_OUT:',df['C4_TOT_OUT'].mean())
#print('min_C4_TOT_OUT:',df['C4_TOT_OUT'].min())
#print('max_C4_TOT_OUT:',df['C4_TOT_OUT'].max())
#print('std_C4_TOT_OUT:',df['C4_TOT_OUT'].std())


#df['nC4_OUT'] = df['nC4_OUT'].interpolate()
#df.plot(y=["nC4_OUT"], sharey=True, kind='line')
#plt.savefig('nC4_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['nC4_OUT'] > 8000]))
#print(df[df['nC4_OUT'] > 8000].head())
#df['nC4_OUT'] = np.array(df['nC4_OUT'].tolist())
#df['nC4_OUT'] = np.where(df['nC4_OUT'] > 8000, 8000 , df['nC4_OUT']).tolist()
#df.plot(y=["nC4_OUT"], sharey=True, kind='line')
#plt.savefig('nC4_OUT-interpolated_a.png')
#print('mean_nC4_OUT:',df['nC4_OUT'].mean())
#print('min_nC4_OUT:',df['nC4_OUT'].min())
#print('max_nC4_OUT:',df['nC4_OUT'].max())
#print('std_nC4_OUT:',df['nC4_OUT'].std())


#df['C5_TOT_OUT'] = df['C5_TOT_OUT'].interpolate()
#df.plot(y=["C5_TOT_OUT"], sharey=True, kind='line')
#plt.savefig('C5_TOT_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['C5_TOT_OUT'] > 3000]))
#print(df[df['C5_TOT_OUT'] > 3000].head())
#print('mean_C5_TOT_OUT:',df['C5_TOT_OUT'].mean())
#print('min_C5_TOT_OUT:',df['C5_TOT_OUT'].min())
#print('max_C5_TOT_OUT:',df['C5_TOT_OUT'].max())
#print('std_C5_TOT_OUT:',df['C5_TOT_OUT'].std())


#df['iC5_OUT'] = df['iC5_OUT'].interpolate()
#df.plot(y=["iC5_OUT"], sharey=True, kind='line')
#plt.savefig('iC5_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['iC5_OUT'] > 1000]))
#print(df[df['iC5_OUT'] > 1000].head())
#print('mean_iC5_OUT:',df['iC5_OUT'].mean())
#print('min_iC5_OUT:',df['iC5_OUT'].min())
#print('max_iC5_OUT:',df['iC5_OUT'].max())
#print('std_iC5_OUT:',df['iC5_OUT'].std())


#df['nC5_OUT'] = df['nC5_OUT'].interpolate()
#df.plot(y=["nC5_OUT"], sharey=True, kind='line')
#plt.savefig('nC5_OUT-interpolated1_a.png')
#plt.show()
#print(len(df[df['nC5_OUT'] > 2000]))
#print(df[df['nC5_OUT'] > 2000].head())
#print('mean_nC5_OUT:',df['nC5_OUT'].mean())
#print('min_nC5_OUT:',df['nC5_OUT'].min())
#print('max_nC5_OUT:',df['nC5_OUT'].max())
#print('std_nC5_OUT:',df['nC5_OUT'].std())


#df['CVD_FLOW'] = df['CVD_FLOW'].interpolate()
#df.plot(y=["CVD_FLOW"], sharey=True, kind='line')
#plt.savefig('CVD_FLOW-interpolated1_a.png')
#plt.show()
#print(len(df[df['CVD_FLOW'] > 0.5]))
#print(df[df['CVD_FLOW'] > 0.5].head())
#print('mean_CVD_FLOW:',df['CVD_FLOW'].mean())
#print('min_CVD_FLOW:',df['CVD_FLOW'].min())
#print('max_CVD_FLOW:',df['CVD_FLOW'].max())
#print('std_CVD_FLOW:',df['CVD_FLOW'].std())



#df['GDS_FLOW'] = df['GDS_FLOW'].interpolate()
#df.plot(y=["GDS_FLOW"], sharey=True, kind='line')
#plt.show()
#plt.savefig('GDS_FLOW-interpolated1_a.png')
#plt.show()
#print(len(df[df['GDS_FLOW'] > 4.0]))
#print(df[df['GDS_FLOW'] > 4.0].head())
#print('mean_GDS_FLOW:',df['GDS_FLOW'].mean())
#print('min_GDS_FLOW:',df['GDS_FLOW'].min())
#print('max_GDS_FLOW:',df['GDS_FLOW'].max())
#print('std_GDS_FLOW:',df['GDS_FLOW'].std())

#df['ROP'] = df['ROP'].interpolate(method='polynomial', order=2)
#df.plot(y=["ROP"], sharey=True, kind='line')
#plt.savefig('ROP-interpolated1_a.png')
#plt.show()
#print(len(df[df['ROP'] > 200]))
#print(df[df['ROP'] > 200].head())
#ma = df['ROP'].max()
#s = df['ROP'].std()
#m = df['ROP'].mean()
#mad = df['ROP'].mad()
#am = abs(m)
#k = 1
#print(am)
#h = (ma + am)/2 + k * mad
#print('h:', h)

#df['ROP'].rolling(window= 3).mean().plot()
#df['ROP'].rolling(window= 3).median().plot(color = 'red')
#df['ROP'].rolling(window= 3).apply(u_t).plot(color = 'cyan')
#df['ROP'].rolling(window= 3).apply(l_t).plot(color ='green')
#plt.savefig('ROP-avg_1a.png')
#plt.show()

#print('mean_ROP:',df['ROP'].mean())
#print('min_ROP:',df['ROP'].min())
#print('max_ROP:',df['ROP'].max())
#print('std_ROP:',df['ROP'].std())


#df['WOB'] = df['WOB'].interpolate(method='polynomial', order=2)
#df.plot(y=["WOB"], sharey=True, kind='line')
#plt.savefig('WOB-interpolated1_a.png')
#plt.show()
#print(len(df[df['WOB'] > 25]))
#print(df[df['WOB'] > 25].head())
#df.plot(y=["WOB"], sharey=True, kind='line')
#df['WOB'].rolling(window= 3).mean().plot()
#df['WOB'].rolling(window= 3).median().plot(color='red')
#df['WOB'].rolling(window=3).apply(u_t).plot(color='cyan')
#df['WOB'].rolling(window=3).apply(l_t).plot(color='green')
#plt.savefig('WOB-avg_1_a.png')
#plt.show()
#print('mean_WOB:',df['WOB'].mean())
#print('min_WOB:',df['WOB'].min())
#print('max_WOB:',df['WOB'].max())
#print('std_WOB:',df['WOB'].std())


#df['RPM'] = df['RPM'].interpolate(method='polynomial', order=2)
#df.plot(y=["RPM"], sharey=True, kind='line')
#plt.savefig('RPM-interpolated1_a.png')
#plt.show()
#print(len(df[df['RPM'] < 80]))
#print(df[df['RPM'] < 80].head())
#ma = df['RPM'].max()
#s = df['RPM'].std()
#m = df['RPM'].mean()
#mad = df['RPM'].mad()
#am = abs(m)
#k = 1
#print(am)
#h = (ma + am)/2 + k * mad
#print('h:', h)

#df['RPM'] = np.where(df['RPM'] > h, h, df['RPM']).tolist()
#df.plot(y=["RPM"], sharey=True, kind='line')
#df['RPM'].rolling(window=3).mean().plot()
#df['RPM'].rolling(window=3).median().plot(color='red')
#df['RPM'].rolling(window=3).apply(u_t).plot(color='cyan')
#df['RPM'].rolling(window=3).apply(l_t).plot(color='green')
#plt.savefig('RPM-avg1_a.png')
#plt.show()
#print('mean_RPM:',df['RPM'].mean())
#print('min_RPM:',df['RPM'].min())
#print('max_RPM:',df['RPM'].max())
#print('std_RPM:',df['RPM'].std())


#df['TORQUE_MAX'] = df['TORQUE_MAX'].interpolate()
#df.plot(y=["TORQUE_MAX"], sharey=True, kind='line')
#plt.savefig('TORQUE_MAX-interpolated1_a.png')
#plt.show()
#print(len(df[df['TORQUE_MAX'] < 10000]))
#print(df[df['TORQUE_MAX'] < 10000].head())
#print('mean_TORQUE_MAX:',df['TORQUE_MAX'].mean())
#print('min_TORQUE_MAX:',df['TORQUE_MAX'].min())
#print('max_TORQUE_MAX:',df['TORQUE_MAX'].max())
#print('std_TORQUE_MAX:',df['TORQUE_MAX'].std())


#df['TORQUE'] = df['TORQUE'].interpolate()
#df.plot(y=["TORQUE"], sharey=True, kind='line')
#plt.savefig('TORQUE-interpolated1_a.png')
#plt.show()
#ma = df['TORQUE'].max()
#s = df['TORQUE'].std()
#m = df['TORQUE'].mean()
#mad = df['TORQUE'].mad()
#am = abs(m)
#k = 1
#print(am)
#h = (ma + am)/2 + k * mad
#print('h:', h)
#df['TORQUE'].rolling(window=3).mean().plot()
#df['TORQUE'].rolling(window=3).median().plot(color='red')
#df['TORQUE'].rolling(window=3).apply(u_t).plot(color='cyan')
#df['TORQUE'].rolling(window=3).apply(l_t).plot(color='green')
#plt.savefig('TORQUE-avg_1a.png')
#plt.show()
#print(len(df[df['TORQUE'] < 10000]))
#print(df[df['TORQUE'] < 10000].head())
#print('mean_TORQUE:',df['TORQUE'].mean())
#print('min_TORQUE:',df['TORQUE'].min())
#print('max_TORQUE:',df['TORQUE'].max())
#print('std_TORQUE:',df['TORQUE'].std())


#df['TORQUE_MIN'] = df['TORQUE_MIN'].interpolate()
#df.plot(y=["TORQUE_MIN"], sharey=True, kind='line')
#plt.savefig('TORQUE_MIN-interpolated1_a.png')
#plt.show()
#print(len(df[df['TORQUE_MIN'] > 20000]))
#print(df[df['TORQUE_MIN'] > 20000].head())
#print('mean_TORQUE_MIN:',df['TORQUE_MIN'].mean())
#print('min_TORQUE_MIN:',df['TORQUE_MIN'].min())
#print('max_TORQUE_MIN:',df['TORQUE_MIN'].max())
#print('std_TORQUE_MIN:',df['TORQUE_MIN'].std())


#df['MW_IN'] = df['MW_IN'].interpolate()
#df.plot(y=["MW_IN"], sharey=True, kind='line')
#plt.savefig('MW_IN-interpolated1_a.png')
#plt.show()
#print(len(df[df['MW_IN'] < 0.5]))
#print(df[df['MW_IN'] < 0.5].head())
#print('mean_MW_IN:',df['MW_IN'].mean())
#print('min_MW_IN:',df['MW_IN'].min())
#print('max_MW_IN:',df['MW_IN'].max())
#print('std_MW_IN:',df['MW_IN'].std())
#df['MW_IN'].rolling(window=30).apply(general_trend).plot()
#df['MW_IN'].rolling(window=30).apply(u_t).plot()
#df['MW_IN'].rolling(window=30).apply(l_t).plot()
#plt.savefig('MW_IN-nonspike_a.png')
#plt.show()

#df['SPP'] = df['SPP'].interpolate()
#df.plot(y=["SPP"], sharey=True, kind='line')
#plt.savefig('SPP-interpolated1_a.png')
#plt.show()
#print('mean_SPP:',df['SPP'].mean())
#print('min_SPP:',df['SPP'].min())
#print('max_SPP:',df['SPP'].max())
#print('std_SPP:',df['SPP'].std())
#df['SPP'].rolling(window=3).mean().plot()
#df['SPP'].rolling(window=3).median().plot(color='red')
#df['SPP'].rolling(window=3).apply(u_t).plot(color='cyan')
#df['SPP'].rolling(window=3).apply(l_t).plot()
#plt.savefig('SPP_avg_1a.png')
#plt.show()

#df['SPP'] = np.where(df['SPP'] > h, h, df['SPP']).tolist()
#df.plot(y=["SPP"], sharey=True, kind='line')
#df['SPP'].rolling(window=5).median().plot()
#df['SPP'].rolling(window=5).std().plot()
#plt.savefig('SPP_nonspike_a.png')
#plt.show()
#df['STEP'] = df['STEP'].interpolate()
#df.plot(y=["STEP"], sharey=True, kind='line')
#plt.savefig('STEP-interpolated1_a.png')
#plt.show()
#print('mean_STEP:',df['STEP'].mean())
#print('min_STEP:',df['STEP'].min())
#print('max_STEP:',df['STEP'].max())
#print('std_STEP:',df['STEP'].std())


#df['ETHYLENE'] = df['ETHYLENE'].interpolate()
#df.plot(y=["ETHYLENE"], sharey=True, kind='line')
#plt.savefig('ETHYLENE-interpolated1_a.png')
#plt.show()
#print('mean_ETHYLENE:',df['ETHYLENE'].mean())
#print('min_ETHYLENE:',df['ETHYLENE'].min())
#print('max_ETHYLENE:',df['ETHYLENE'].max())
#print('std_ETHYLENE:',df['ETHYLENE'].std())


#df['PROPYLENE'] = df['PROPYLENE'].interpolate()
#df.plot(y=["PROPYLENE"], sharey=True, kind='line')
#plt.savefig('PROPYLENE-interpolated1_a.png')
#plt.show()
#print('mean_PROPYLENE:',df['PROPYLENE'].mean())
#print('min_PROPYLENE:',df['PROPYLENE'].min())
#print('max_PROPYLENE:',df['PROPYLENE'].max())
#print('std_PROPYLENE:',df['PROPYLENE'].std())

#df.plot(y=["C1_OUT","C2_OUT","C3_OUT","iC4_OUT","C4_TOT_OUT","nC4_OUT","C5_TOT_OUT","iC5_OUT","nC5_OUT"], subplots=True, layout=(3,3), kind='line')
#plt.savefig('c1,c2,c3,c4,c5interpolated_1_a.png')
#plt.show()







#df = df.replace(to_replace=np.nan, value=0, method='pad')

#Scaling data and Normalization
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(df)
#df = pd.DataFrame(x_scaled)
#print(df)















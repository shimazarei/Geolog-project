import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas as pd

columns = ['Index', 'MEASURED_DEPTH', 'STEP', 'TIME_DRILLING', 'VERTICAL_DEPTH', 'TOT_GAS', 'C1_OUT', 'C2_OUT', 'C3_OUT', 'iC4_OUT', 'C4_TOT_OUT', 'nC4_OUT', 'ROP', 'WOB', 'RPM', 'iC5_OUT', 'C5_TOT_OUT','nC5_OUT','CVD_FLOW','GDS_FLOW','TORQUE_MAX','TORQUE','TORQUE_MIN','MW_IN','SPP','ETHYLENE','PROPYLENE']
raw_data = pd.read_csv("well_1_a.csv", index_col="Index")
df = pd.DataFrame(raw_data, columns = ['Index', 'MEASURED_DEPTH', 'STEP', 'TIME_DRILLING', 'VERTICAL_DEPTH', 'TOT_GAS', 'C1_OUT', 'C2_OUT', 'C3_OUT', 'iC4_OUT', 'C4_TOT_OUT', 'nC4_OUT', 'ROP', 'WOB', 'RPM', 'iC5_OUT', 'C5_TOT_OUT','nC5_OUT','CVD_FLOW','GDS_FLOW','TORQUE_MAX','TORQUE','TORQUE_MIN','MW_IN','SPP','ETHYLENE','PROPYLENE']
)
df = df.drop(df.index[0])
df = df[columns].apply(pd.to_numeric)
df[df < 0] = np.nan

df = df.replace(to_replace=np.nan, value=0, method='pad')

df['ROP'] = df['ROP'].replace(to_replace=0, value=np.mean(df['ROP']), method='pad')

df['RPM'] = df['RPM'].replace(to_replace=0, value=np.median(df['RPM']), method='pad')

df['TOT_GAS'] = df['TOT_GAS'].replace(to_replace=0, value=np.mean(df['TOT_GAS']), method='pad')

df['SPP'] = df['SPP'].replace(to_replace=0, value=np.mean(df['SPP']), method='pad')

df['WOB'] = df['WOB'].replace(to_replace=0, value=np.mean(df['WOB']), method='pad')

df['TORQUE'] = df['TORQUE'].replace(to_replace=0, value=np.mean(df['TORQUE']), method='pad')

df['STEP'] = df['STEP'].replace(to_replace=0, value=np.mean(df['STEP']), method='pad')

df['TIME_DRILLING'] = df['TIME_DRILLING'].replace(to_replace=0, value=np.mean(df['TIME_DRILLING']), method='pad')

df['MEASURED_DEPTH'] = df['MEASURED_DEPTH'].replace(to_replace=0, value=np.mean(df['MEASURED_DEPTH']), method='pad')

df['C1_OUT'] = df['C1_OUT'].replace(to_replace=0, value=np.mean(df['C1_OUT']), method='pad')

df['C2_OUT'] = df['C2_OUT'].replace(to_replace=0, value=np.mean(df['C2_OUT']), method='pad')

df['C3_OUT'] = df['C3_OUT'].replace(to_replace=0, value=np.mean(df['C3_OUT']), method='pad')

df['iC4_OUT'] = df['iC4_OUT'].replace(to_replace=0, value=np.mean(df['iC4_OUT']), method='pad')

df['C4_TOT_OUT'] = df['C4_TOT_OUT'].replace(to_replace=0, value=np.mean(df['C4_TOT_OUT']), method='pad')

df['nC4_OUT'] = df['nC4_OUT'].replace(to_replace=0, value=np.mean(df['nC4_OUT']), method='pad')

df['iC5_OUT'] = df['iC5_OUT'].replace(to_replace=0, value=np.mean(df['iC5_OUT']), method='pad')

df['C5_TOT_OUT'] = df['C5_TOT_OUT'].replace(to_replace=0, value=np.mean(df['C5_TOT_OUT']), method='pad')

df['nC5_OUT'] = df['nC5_OUT'].replace(to_replace=0, value=np.mean(df['nC5_OUT']), method='pad')

df['CVD_FLOW'] = df['CVD_FLOW'].replace(to_replace=0, value=np.mean(df['CVD_FLOW']), method='pad')

df['GDS_FLOW'] = df['GDS_FLOW'].replace(to_replace=0, value=np.mean(df['GDS_FLOW']), method='pad')

df['TORQUE_MAX'] = df['TORQUE_MAX'].replace(to_replace=0, value=np.mean(df['TORQUE_MAX']), method='pad')

df['TORQUE_MIN'] = df['TORQUE_MIN'].replace(to_replace=0, value=np.mean(df['TORQUE_MAX']), method='pad')

df['MW_IN'] = df['MW_IN'].replace(to_replace=0, value=np.mean(df['MW_IN']), method='pad')

df['ETHYLENE'] = df['ETHYLENE'].replace(to_replace=0, value=np.mean(df['ETHYLENE']), method='pad')

df['PROPYLENE'] = df['PROPYLENE'].replace(to_replace=0, value=np.mean(df['PROPYLENE']), method='pad')


def Threshold_algorithm(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    medianFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    medianFilter[lag - 1] = np.median(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    uppperthr = medianFilter + threshold * stdFilter
    lowerthr = np.array(medianFilter) - np.array(threshold) * stdFilter
    for i in range(lag, len(y) - 1):
        if abs(y[i] - medianFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > uppperthr[i - 1]:
                signals[i] = 1
            else:
              if abs(y[i] - medianFilter[i - 1]) < threshold * stdFilter[i - 1]:
                if y[i] < lowerthr[i - 1]:
                     signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            medianFilter[i] = np.median(filteredY[(i - lag):i])
            stdFilter[i] = np.std(filteredY[(i - lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            medianFilter[i] = np.median(filteredY[(i - lag):i])
            stdFilter[i] = np.std(filteredY[(i - lag):i])

    return dict(signals=np.asarray(signals),
                    medianFilter=np.asarray(medianFilter),
                    stdFilter=np.asarray(stdFilter))


lag =50
threshold = 5
influence = 0.5
y = df['SPP']

result  = Threshold_algorithm(y = y, lag = lag, threshold = threshold, influence = influence)

pylab.subplot(211)
pylab.plot(np.arange(1, len(y)+1), y)

pylab.plot(np.arange(1, len(y)+1),
           result["medianFilter"], color="cyan", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["medianFilter"] + threshold * result["stdFilter"], color="orange", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["medianFilter"] - threshold * result["stdFilter"], color="green", lw=2)

pylab.subplot(212)
pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.ylim(-1.5, 1.5)
#pylab.savefig('TOT_GAS9ethrspike.png')
#pylab.savefig('ROP_1athrspike.png')
#pylab.savefig('WOB-9ethrspike.png')
#pylab.savefig('RPM_9ethrspike.png')
#pylab.savefig('SPP_9ethrspike.png')
#pylab.savefig('TORQUE_9ethrspike.png')
plt.show()



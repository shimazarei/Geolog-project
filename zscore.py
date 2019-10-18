import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

columns = ['Index', 'MEASURED_DEPTH', 'STEP', 'TIME_DRILLING', 'VERTICAL_DEPTH', 'TOT_GAS', 'C1_OUT', 'C2_OUT', 'C3_OUT', 'iC4_OUT', 'C4_TOT_OUT', 'nC4_OUT', 'ROP', 'WOB', 'RPM', 'iC5_OUT', 'C5_TOT_OUT','nC5_OUT','CVD_FLOW','GDS_FLOW','TORQUE_MAX','TORQUE','TORQUE_MIN','MW_IN','SPP','ETHYLENE','PROPYLENE']
raw_data = pd.read_csv("well_1_a.csv", index_col="Index")
df = pd.DataFrame(raw_data, columns = ['Index', 'MEASURED_DEPTH', 'STEP', 'TIME_DRILLING', 'VERTICAL_DEPTH', 'TOT_GAS', 'C1_OUT', 'C2_OUT', 'C3_OUT', 'iC4_OUT', 'C4_TOT_OUT', 'nC4_OUT', 'ROP', 'WOB', 'RPM', 'iC5_OUT', 'C5_TOT_OUT','nC5_OUT','CVD_FLOW','GDS_FLOW','TORQUE_MAX','TORQUE','TORQUE_MIN','MW_IN','SPP','ETHYLENE','PROPYLENE']
)
df = df.drop(df.index[0])
df = df[columns].apply(pd.to_numeric)
df[df < 0] = np.nan

mean1 = np.mean(df['ROP'])
mean2 = np.mean(df['RPM'])
mean3 = np.mean(df['TOT_GAS'])
mean4 = np.mean(df['SPP'])
mean5 = np.mean(df['WOB'])
mean6 = np.mean(df['TORQUE'])


df['ROP'] = df['ROP'].replace(to_replace=np.nan, value=mean1, method='pad')
df['ROP'] = df['ROP'].replace(to_replace=0, value=mean1, method='pad')

df['RPM'] = df['RPM'].replace(to_replace=np.nan, value=mean2, method='pad')
df['RPM'] = df['RPM'].replace(to_replace=0, value=mean2, method='pad')

df['TOT_GAS'] = df['TOT_GAS'].replace(to_replace=np.nan, value=mean3, method='pad')
df['TOT_GAS'] = df['TOT_GAS'].replace(to_replace=0, value=mean3, method='pad')

df['SPP'] = df['SPP'].replace(to_replace=np.nan, value=mean4, method='pad')
df['SPP'] = df['SPP'].replace(to_replace=0, value=mean4, method='pad')

df['WOB'] = df['WOB'].replace(to_replace=np.nan, value=mean5, method='pad')
df['WOB'] = df['WOB'].replace(to_replace=0, value=mean5, method='pad')

df['TORQUE'] = df['TORQUE'].replace(to_replace=np.nan, value=mean6, method='pad')
df['TORQUE'] = df['TORQUE'].replace(to_replace=0, value=mean6, method='pad')





def ZScore_algorithm(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            avgFilter[i] = np.mean(filteredY[(i - lag):i])
            stdFilter[i] = np.std(filteredY[(i - lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag):i])
            stdFilter[i] = np.std(filteredY[(i - lag):i])

    return dict(signals=np.asarray(signals),
                    avgFilter=np.asarray(avgFilter),
                    stdFilter=np.asarray(stdFilter))


lag =50
threshold =5
influence = 0.5
y = df['ROP']
result  = ZScore_algorithm(y = y, lag = lag, threshold = threshold, influence = influence)
print(result)

pylab.subplot(211)
pylab.plot(np.arange(1, len(y)+1), y)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"], color="cyan", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"] + threshold * result["stdFilter"], color="orange", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

pylab.subplot(212)
pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.ylim(-1.5, 1.5)
#pylab.savefig('TOT_GAS9e-zscore.png')
#pylab.savefig('ROP10f-zscore.png')
#pylab.savefig('RPM9e-zscore.png')
#pylab.savefig('WOB9e-zscore.png')
#pylab.savefig('SPP9e-zscore.png')
#pylab.savefig('TORQUE9e-zscore.png')
plt.show()

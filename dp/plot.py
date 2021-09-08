import matplotlib.pyplot as plt
import numpy as np


metrics = ['acc', 'f1 score', 'auroc', 'auprc']
plt.figure(figsize=(10, 5))
X = np.arange(4)

# plt.title("Adult Dataset")
# plt.bar(X + 0.00, [0.8238, 0.8854, 0.8585, 0.9385], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.7151, 0.8275, 0.6739 ,0.8511], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.7359, 0.8456, 0.5097, 0.7653], width=0.25, color='#ED8975')

# plt.title("Breast Cancer Dataset")
# plt.bar(X + 0.00, [0.9591, 0.9675, 0.9809, 0.9831], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.4459, 0.2760, 0.6820, 0.7619], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.5629, 0.5973, 0.6601, 0.7377], width=0.25, color='#ED8975')

# plt.title("Cardio Dataset")
# plt.bar(X + 0.00, [0.6620, 0.6872, 0.7149, 0.6960], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.5046, 0.1237, 0.4958, 0.5072], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.5583, 0.2971, 0.6827, 0.6709], width=0.25, color='#ED8975')

# plt.title("Cervical Dataset")
# plt.bar(X + 0.00, [0.9516, 0.5480, 0.8218, 0.5325], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.9331, 0.1240, 0.6412, 0.1714], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.9302, 0.0685, 0.5084, 0.1325], width=0.25, color='#ED8975')

# plt.title("Covertype Dataset")
# plt.bar(X + 0.00, [0.6718, 0.6495, 0.7759, 0.11], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.4034, 0.2675, 0.5339, 1], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.6947, 0.6773, 0.7722, 0], width=0.25, color='#ED8975')
# plt.xticks(X + 0.25, ['acc', 'f1 score', 'auroc', 'normalized log loss'])

# plt.title("ilpd Dataset")
# plt.bar(X + 0.00, [0.6681, 0.3048, 0.7230, 0.5601], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.6408, 0.0143, 0.6050, 0.4473], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.6408, 0.0452, 0.5983, 0.4299], width=0.25, color='#ED8975')

# plt.title("Seizure Dataset")
# plt.bar(X + 0.00, [0.8926, 0.6542, 0.7897, 0.7201], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.7762, 0.1974, 0.4912, 0.4017], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.7657, 0.2101, 0.5, 0.4086], width=0.25, color='#ED8975')

# plt.title("Adult Dataset")
# plt.bar(X + 0.00, [0.7359, 0.8456, 0.5097, 0.7653], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.6254, 0.6433, 0.5008, 0.7500], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.6257, 0.6434, 0.5189, 0.7633], width=0.25, color='#ED8975')

# plt.title("Cardio Dataset")
# plt.bar(X + 0.00, [0.5583, 0.2971, 0.6827, 0.6709], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.4991, 0.0, 0.4648, 0.4825], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.4938, 0.4391, 0.4876, 0.4987], width=0.25, color='#ED8975')

# plt.title("Cervical Dataset")
# plt.bar(X + 0.00, [0.9302, 0.0685, 0.5084, 0.1325], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.7180, 0.0301, 0.4920, 0.1343], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.7180, 0.0301, 0.4617, 0.0776], width=0.25, color='#ED8975')

# plt.title("Covertype Dataset")
# plt.bar(X + 0.00, [0.6947, 0.6773, 0.7722, 0], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.2364, 0.1524, 0.487, 1], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.4215, 0.2997, 0.4851, 0.3491], width=0.25, color='#ED8975')
# plt.xticks(X + 0.25, ['acc', 'f1 score', 'auroc', 'normalized log loss'])

# plt.title("Credit Card Dataset")
# plt.bar(X + 0.00, [0.7167, 0.0160, 0.7405, 0.1553], width=0.25, color='#8FB9AA')
# plt.bar(X + 0.25, [0.8893, 0.0041, 0.5619, 0.0117], width=0.25, color='#F2D096')
# plt.bar(X + 0.50, [0.9959, 0.0, 0.5707, 0.0044], width=0.25, color='#ED8975')

plt.title("Seizure Dataset")
plt.bar(X + 0.00, [0.7657, 0.2101, 0.5, 0.4086], width=0.25, color='#8FB9AA')
plt.bar(X + 0.25, [0.5991, 0.2118, 0.4869, 0.2000], width=0.25, color='#F2D096')
plt.bar(X + 0.50, [0.6836, 0.1752, 0.4902, 0.1992], width=0.25, color='#ED8975')

plt.xticks(X + 0.25, metrics)
plt.legend(['DP-CTGAN', 'PATE-GAN', 'DP-GAN'], loc='lower right')

plt.show()

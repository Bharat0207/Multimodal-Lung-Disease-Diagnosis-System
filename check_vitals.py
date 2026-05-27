import numpy as np

vitals = np.load(r"D:\mimic_project\dataset\vitals_timeseries.npy")

print("Shape:", vitals.shape)

# count non-zero entries before normalization assumption
nonzero = np.count_nonzero(vitals)

total = vitals.size

print("\nNon-zero values:", nonzero)
print("Total values:", total)
print("Coverage:", round(nonzero/total,4))

# check a random patient
import random
p = random.randint(0, vitals.shape[0]-1)

print("\nRandom patient sample:")
print(vitals[p][:10])
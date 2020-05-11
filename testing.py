from data_utils import *
from utils import *

load_data(verbose=True)

arr=np.array([1,2])

sens=calculate_regularized_sensitivity(10000,0.25, 5, 2, 1)
noisy_arr=add_noise(arr, sens, 1)

print(arr)
print(noisy_arr)


bsense= calculate_unregularized_sensitivity(2, 2, 1)
noisy_arr=add_noise(arr, sens, 1)

print(arr)
print(noisy_arr)

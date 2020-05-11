from data_utils import *
from utils import *

x,y=load_data()
x_train, x_test, y_train, y_test=test_split(x, y)

# rand=np.random.RandomState(1000)
original_model=train_svm_reg(x_train, y_train, 0.01)
single_model_performance(x_test, y_test,original_model)
private_model=private_model(x_train, y_train, original_model, 0.01, 0.01)
single_model_performance(x_test, y_test,private_model)

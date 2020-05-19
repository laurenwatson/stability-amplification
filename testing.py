from data_utils import *
from utils import *

x,y=load_data('har', True)
x_train, x_test, y_train, y_test=test_split(x, y)

# # rand=np.random.RandomState(1000)
original_model=train_svm_reg(x_train, y_train, 0.01)
single_model_performance(x_test, y_test,original_model)
private_model=private_model(x_train, y_train, original_model, 0.01, 0.1)
single_model_performance(x_test, y_test,private_model)

plot_reg_curve(original_model, private_model,'Model performance', x_train, y_train, x_test, y_test, 0.01, 0.1)

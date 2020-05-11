# Privacy Amplification by Stability

### Description

 Experiments for 'Privacy Amplification by Stability', using  stochastic gradient descent for convex empirical risk minimization learning problems with 'bolt-on' differential privacy via output perturbation.


 ### Data
The data included here is:
- The *Human Activity Recognition Using Smartphones* dataset, from the UCI Machine Learning Repository.
- The *Adult* dataset, from the UCI Machine Learning Repository.


 ### Models

 ### Differential Privacy
This repository is not intended to be an end-to-end implementation of a fully differentially private learning _system_. The output of each individual model trained is differentially private, due to output perturbation of the final learned weights.

This work aims to provide empirical evidence of the performance of differentially private models, in order to validate theoretical work. If used practice, each training run used to obtain the average performance for each model would contribute to an overall privacy budget and other considerations such as differentially private hyperparameter tuning would be needed.


 ### Running Experiments

 #### Requirements
 - Python 3.6.10
 - Matplotlib 3.1.3
 - Pandas 1.0.3
 - Numpy 1.18.1

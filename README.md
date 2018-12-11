# Envision

This is my internship project in Envision.

The goal of this project is to predict the wind speed of turbines.

There is the log of this project:

Week 1

Visualization on data to explore more information. mainly include: plot the distribution of wind speed and

the relationship between power and wind speed. besides, build a classifier to explore difference of different range

of wind speed.

Week2

Build single weather source regression model and grid search to find best parameters.

Week3

Do data pre-processing, such as smooth target attributes Y. do more feature engineering, including splitting

month into season and changing numeric value (wind direction) into category attributes. build multiple weather sources

model.

Week4

Add outlier check code to delete some potential data. try to use sequence model to capture sequence information

among data. build ARMA model and test the performance.

Week5

Modify and debug code to improve the performance, and complete stage 1 report and presentation slides.

Week6

Start to modify real project code. build a linear model as the second layer, including simple linear model,

Bridge regression, lasso regression, ElasticNet model. also build some non-linear models, such as SVR and Random Forest.

Week7

Add a new non-linear model as the second layer. try to build different models with different horizons.

add RMSE testing function. focus on data 3-15m/s wind speed and add new filter code.

Week8

Use data of new farm WF0010 to retest the performance of models. modify and debug code.

Week9

Test four linear models on data of two farms. choose to Bridge and ElasticNet regression and submit code.

Week10

Add simple neural network - fully connected as the second layer to improve the overall performance of the model.

and submit code.

Week11

Use RNN (LSTM) replacing XGB to capture sequence information at the first layer. the time of training model is

too large with normal CPU. the results of RNN and linear model is not bad.

Week12

Complete stage 2 report and submit code.
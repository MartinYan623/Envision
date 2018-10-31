## A machine learning model that is used to map the forecast wind speed to the turbine wind speed

For each weather source, the mapping is learned independently and built on a xgb model. 

The usage for each script
* main class file `xgb_ws_forecast.py`
* model training file `train_ws_model.py`. input: raw weather forecast data, output: the trained model, and feature file by replacing the wind speed to the revised one for each weather source. 
* testing data generation file `generate_ws_data.py`. input: raw weather forecast data for the testing periods. output: feature files by replacing the wind speed to the revised one based on the trained model. 

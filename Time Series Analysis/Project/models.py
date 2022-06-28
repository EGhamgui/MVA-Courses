import numpy as np 
import pandas as pd 
import plotly.graph_objs as go
import matplotlib.pyplot as plt 
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
from keras.layers import Dense, LSTM, Flatten, Dropout

import warnings
warnings.filterwarnings("ignore")

#######################################################################################

def RF_regression(X_train, y_train, X_test, y_test, date, size1, plot=True) :

  # Create the model 
  RF = RandomForestRegressor(random_state=0)

  # Time series separation for data 
  tscv = TimeSeriesSplit(n_splits=5)

  # Define the dictionary for Grid Search 
  p_grid_RF = {'n_estimators': [10,15,20,25,30,40,50,75,100], 'min_samples_leaf': [2,3,4,5,6], 'max_features': ['sqrt','log2']} 

  # Perform Grid Search 
  grid_RF = GridSearchCV(estimator=RF, param_grid=p_grid_RF, scoring='neg_mean_absolute_error', cv=tscv)
  grid_RF.fit(X_train, y_train)

  if plot : 
    print("Best Validation Score: {}".format(-1 * grid_RF.best_score_))
    print("Best params: {}".format(grid_RF.best_params_))
  
  # Define the model 
  RF_reg = RandomForestRegressor(n_estimators=grid_RF.best_params_['n_estimators'], 
           min_samples_split=grid_RF.best_params_['min_samples_leaf'],max_features=grid_RF.best_params_['max_features']) 

  # Train the model 
  RF_reg.fit(X_train,y_train)

  # Predict test data 
  y_pred = RF_reg.predict(X_test)

  dd = np.r_[y_train.reshape(-1,1),np.array(y_pred).reshape(-1,1)]
  dd = pd.DataFrame(dd, index=date, columns=['next_day_ret'])
  data2 = dd[size1-1:size1+1]

  # chart
  if plot :
    fig = make_subplots(rows=1, cols=1,subplot_titles =['Next day return (train + predictions)'])
    fig.add_trace(go.Scatter(x=dd.index[:size1], y=dd['next_day_ret'][:size1], marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
    fig.add_trace(go.Scatter(x=data2.index, y=data2['next_day_ret'], marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8', mode='lines'),row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index[size1:], y=dd['next_day_ret'][size1:], marker={'color':'#FC5A50'}, fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
    fig.update_layout(showlegend=False)
    fig.show()

  # Compute metrics 
  mae = mean_absolute_error(y_test, y_pred) 
  rmse = mean_squared_error(y_test, y_pred,squared=False)
  mape = mean_absolute_percentage_error(y_test, y_pred) 

  if plot : 
    print('Mean Absolute Error on test data:',round(mae,3))
    print('Root Mean Squared Error on test data:',round(rmse,3))
    print('Mean Absolute Percentage Error on test data:',round(mape,3))

  # Error plot 
  if plot : 
    error = np.abs(y_test.reshape(-1,) - np.array(y_pred).reshape(-1,))
    fig = make_subplots(rows=1, cols=1,subplot_titles =['Error Plot'])
    fig.add_trace(go.Scatter(x=dd.index[size1:],y=error, marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
    fig.update_layout(showlegend=False)
    fig.show()
  
  return mae, rmse, mape

# --------------------------------------------------------------------- #

def GB_regression(X_train, y_train, X_test, y_test, date, size1) :

  # Create the model 
  GB = GradientBoostingRegressor(random_state=0)

  # Define the dictionary for Grid Search 
  p_grid_GB = {'n_estimators': [10,15,20,25,30,40,50,75,100], 'min_samples_leaf': [2,3,4,5,6], 'max_features': ['sqrt','log2']}  

  # Time series separation for data 
  tscv = TimeSeriesSplit(n_splits=5)

  # Perform Grid Search 
  grid_GB = GridSearchCV(estimator=GB, param_grid=p_grid_GB, scoring='neg_mean_absolute_error', cv=tscv)
  grid_GB.fit(X_train, y_train)

  print("Best Validation Score: {}".format(-1 * grid_GB.best_score_))
  print("Best params: {}".format(grid_GB.best_params_))
  
  # Define the model 
  GB_reg = RandomForestRegressor(n_estimators=grid_GB.best_params_['n_estimators'], 
           min_samples_split=grid_GB.best_params_['min_samples_leaf'],max_features=grid_GB.best_params_['max_features']) 

  # Train the model 
  GB_reg.fit(X_train,y_train)

  # Predict test data 
  y_pred = GB_reg.predict(X_test)

  dd = np.r_[y_train.reshape(-1,1),np.array(y_pred).reshape(-1,1)]
  dd = pd.DataFrame(dd, index=date, columns=['next_day_ret'])
  data2 = dd[size1-1:size1+1]

  # chart
  fig = make_subplots(rows=1, cols=1,subplot_titles =['Next day return (train + predictions)'])
  fig.add_trace(go.Scatter(x=dd.index[:size1], y=dd['next_day_ret'][:size1], marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
  fig.add_trace(go.Scatter(x=data2.index, y=data2['next_day_ret'], marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8', mode='lines'),row=1, col=1)
  fig.add_trace(go.Scatter(x=dd.index[size1:], y=dd['next_day_ret'][size1:], marker={'color':'#FC5A50'}, fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
  fig.update_layout(showlegend=False)
  fig.show()

  # Compute metrics 
  mae = mean_absolute_error(y_test, y_pred) 
  rmse = mean_squared_error(y_test, y_pred,squared=False)
  mape = mean_absolute_percentage_error(y_test, y_pred) 

  print('Mean Absolute Error on test data:',round(mae,3))
  print('Root Mean Squared Error on test data:',round(rmse,3))
  print('Mean Absolute Percentage Error on test data:',round(mape,3))

  # Error plot 
  error = np.abs(y_test.reshape(-1,) - np.array(y_pred).reshape(-1,))
  fig = make_subplots(rows=1, cols=1,subplot_titles =['Error Plot'])
  fig.add_trace(go.Scatter(x=dd.index[size1:],y=error, marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
  fig.update_layout(showlegend=False)
  fig.show()

# --------------------------------------------------------------------- #

def NN_regression(X_train, y_train, X_test, y_test, date, size1) :

  # Transform Train data
  X_train_deep = np.asarray(X_train).astype('float32').reshape((X_train.shape[0],X_train.shape[1],1))

  # Create the model 
  multivariate_stacked_lstm = tf.keras.models.Sequential([
      LSTM(100, input_shape=(X_train_deep.shape[1],X_train_deep.shape[2]), return_sequences=True),
      Flatten(),
      Dense(64),
      Dropout(0.5),
      Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, amsgrad=True)
  loss = tf.keras.losses.MeanSquaredError()
  metric = [tf.keras.metrics.RootMeanSquaredError()]
  early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
  multivariate_stacked_lstm.compile(loss=loss,
                                    optimizer=optimizer,
                                    metrics=metric)

  # Train the model 
  multivariate_stacked_lstm.fit(X_train_deep, np.array(y_train) , epochs=120, batch_size= 10, validation_split=0.2,callbacks=[early_stopping])

  # Predict test data 
  y_pred = multivariate_stacked_lstm.predict(np.asarray(X_test).astype('float32'))

  dd = np.r_[y_train.reshape(-1,1),np.array(y_pred).reshape(-1,1)]
  dd = pd.DataFrame(dd, index=date, columns=['next_day_ret'])
  data2 = dd[size1-1:size1+1]

  # chart
  fig = make_subplots(rows=1, cols=1,subplot_titles =['Next day return (train + predictions)'])
  fig.add_trace(go.Scatter(x=dd.index[:size1], y=dd['next_day_ret'][:size1], marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
  fig.add_trace(go.Scatter(x=data2.index, y=data2['next_day_ret'], marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8', mode='lines'),row=1, col=1)
  fig.add_trace(go.Scatter(x=dd.index[size1:], y=dd['next_day_ret'][size1:], marker={'color':'#FC5A50'}, fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
  fig.update_layout(showlegend=False)
  fig.show()

  # Compute metrics 
  mae = mean_absolute_error(y_test, y_pred) 
  rmse = mean_squared_error(y_test, y_pred,squared=False)
  mape = mean_absolute_percentage_error(y_test, y_pred) 

  print('Mean Absolute Error on test data:',round(mae,3))
  print('Root Mean Squared Error on test data:',round(rmse,3))
  print('Mean Absolute Percentage Error on test data:',round(mape,3))

  # Error plot 
  error = np.abs(y_test.reshape(-1,) - np.array(y_pred).reshape(-1,))
  fig = make_subplots(rows=1, cols=1,subplot_titles =['Error Plot'])
  fig.add_trace(go.Scatter(x=dd.index[size1:],y=error, marker={'color':'#496595'},fill='tozeroy', fillcolor='#c6ccd8',mode='lines'),row=1, col=1)
  fig.update_layout(showlegend=False)
  fig.show()
  
# --------------------------------------------------------------------- #

def RF_classification(X_train, y_train, X_test, y_test, show=False) : 

  # Create the model 
  RF = RandomForestClassifier(random_state=0)

  # Define the dictionary for Grid Search 
  p_grid_RF = {'n_estimators': [10,15,20,25,30,40,50,75,100], 'min_samples_leaf': [2,3,4,5,6], 'max_features': ['sqrt','log2']}  

  # Perform Grid Search 
  grid_RF = GridSearchCV(estimator=RF, param_grid=p_grid_RF, cv=5)
  grid_RF.fit(X_train, y_train)

  # Define the model 
  RF_class = RandomForestClassifier(n_estimators=grid_RF.best_params_['n_estimators'], 
           min_samples_split=grid_RF.best_params_['min_samples_leaf'],max_features=grid_RF.best_params_['max_features']) 

  # Train the model 
  RF_class.fit(X_train,y_train)

  # Predict test data 
  y_pred_ = RF_class.predict(X_test)

  # Compute metrics 
  acc = accuracy_score(y_test, y_pred_)
  precision = precision_score(y_test, y_pred_)
  recall = recall_score(y_test, y_pred_)
  f1s = f1_score(y_test, y_pred_)

  if show:
    print("Best Validation Score: {}".format(grid_RF.best_score_))
    print("Best params: {}".format(grid_RF.best_params_))
    print('Accuracy on test data:',round(acc,3))
    print('Precision on test data:',round(precision,3))
    print('Recall on test data:',round(recall,3))
    print('F1-score on test data:',round(f1s,3))

    # Plot Confusion matrix 
    CM = confusion_matrix(y_test, y_pred_)
    CM = np.round(CM.astype('float') / CM.sum(axis=1)[:, np.newaxis],2)
    x = ['Bad','Good']
    y = ['Bad','Good']
    fig = make_subplots(rows=1, cols=2, subplot_titles =['Confusion Matrix', 'Precision-Recall Curve'])
    fig.add_trace(go.Heatmap(x=x,y=y,z=CM,colorscale='Blues',text= CM.astype('str'), texttemplate="%{text}", showscale=False),row=1,col=1)

    # Plot Precision-Recall Curve  
    y_onehot = pd.get_dummies(y_test, columns=RF_class.classes_)
    y_preds = RF_class.predict_proba(X_test)
    columns = ['Bad', 'Good']
    name = ['Bad', 'Good']
    for i in range(y_preds.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_pred = y_preds[:, i]
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auc_score = average_precision_score(y_true, y_pred)
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name[i], mode='lines'),row=1,col=2)

    fig.update_layout(height=500, width=1000)
    fig.show()
    
  return acc, precision, recall, f1s

# --------------------------------------------------------------------- #

def GB_classification(X_train, y_train, X_test, y_test) : 

  # Create the model 
  GB = GradientBoostingClassifier(random_state=0)

  # Define the dictionary for Grid Search 
  p_grid_GB = {'n_estimators': [10,15,20,25,30,40,50,75,100], 'min_samples_leaf': [2,3,4,5,6], 'max_features': ['sqrt','log2']} 

  # Perform Grid Search 
  grid_GB = GridSearchCV(estimator=GB, param_grid=p_grid_GB, cv=5)
  grid_GB.fit(X_train, y_train)

  print("Best Validation Score: {}".format(grid_GB.best_score_))
  print("Best params: {}".format(grid_GB.best_params_))
  
  # Define the model 
  GB_class = RandomForestClassifier(n_estimators=grid_GB.best_params_['n_estimators'], 
           min_samples_split=grid_GB.best_params_['min_samples_leaf'],max_features=grid_GB.best_params_['max_features']) 

  # Train the model 
  GB_class.fit(X_train,y_train)

  # Predict test data 
  y_pred_ = GB_class.predict(X_test)

  # Compute metrics 
  acc = accuracy_score(y_test, y_pred_)
  precision = precision_score(y_test, y_pred_)
  recall = recall_score(y_test, y_pred_)
  f1s = f1_score(y_test, y_pred_)
  print('Accuracy on test data:',round(acc,3))
  print('Precision on test data:',round(precision,3))
  print('Recall on test data:',round(recall,3))
  print('F1-score on test data:',round(f1s,3))

  # Plot Confusion matrix 
  CM = confusion_matrix(y_test, y_pred_)
  CM = np.round(CM.astype('float') / CM.sum(axis=1)[:, np.newaxis],2)
  x = ['Bad','Good']
  y = ['Bad','Good']
  fig = make_subplots(rows=1, cols=2, subplot_titles =['Confusion Matrix', 'Precision-Recall Curve'])
  fig.add_trace(go.Heatmap(x=x,y=y,z=CM,colorscale='Blues',text= CM.astype('str'), texttemplate="%{text}", showscale=False),row=1,col=1)

  # Plot Precision-Recall Curve
  y_onehot = pd.get_dummies(y_test, columns=[0,1])
  y_preds = GB_class.predict_proba(X_test)
  columns = ['Bad', 'Good']
  name = ['Bad', 'Good']
  for i in range(y_preds.shape[1]):
      y_true = y_onehot.iloc[:, i]
      y_pred = y_preds[:, i]
      precision, recall, _ = precision_recall_curve(y_true, y_pred)
      auc_score = average_precision_score(y_true, y_pred)
      fig.add_trace(go.Scatter(x=recall, y=precision, name=name[i], mode='lines'),row=1,col=2)

  fig.update_layout(height=500, width=1000)
  fig.show()

# --------------------------------------------------------------------- #

def NN_classification(X_train, y_train, X_test, y_test) : 
  
  # ------- Define the architecture of the model ------- #
  
  # Initialize the constructor
  model = tf.keras.Sequential()

  # Add an input layer 
  model.add(Dense(128, activation='relu'))

  # Add a hidden layer 
  model.add(Dense(64, activation='relu'))

  # Add a hidden layer 
  model.add(Dense(32, activation='sigmoid'))

  # Add an output layer 
  model.add(Dense(1, activation='sigmoid'))

  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, amsgrad=True)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
                    
  history = model.fit(X_train, y_train, epochs = 30, batch_size=20, verbose=1)
  temp = model.predict(X_test)
  y_pred_ = ( temp > 0.6).astype(int).reshape(-1,1)
  y_preds = np.zeros((len(temp),2))
  y_preds[:,0] = temp[:,0]
  y_preds[:,1] = 1-np.array(temp[:,0])

  # Compute metrics 
  acc = accuracy_score(y_test, y_pred_)
  precision = precision_score(y_test, y_pred_)
  recall = recall_score(y_test, y_pred_)
  f1s = f1_score(y_test, y_pred_)
  print('Accuracy on test data:',round(acc,3))
  print('Precision on test data:',round(precision,3))
  print('Recall on test data:',round(recall,3))
  print('F1-score on test data:',round(f1s,3))

  # Plot Confusion matrix 
  CM = confusion_matrix(y_test, y_pred_)
  CM = np.round(CM.astype('float') / CM.sum(axis=1)[:, np.newaxis],2)
  x = ['Bad','Good']
  y = ['Bad','Good']
  fig = make_subplots(rows=1, cols=2, subplot_titles =['Confusion Matrix', 'Precision-Recall Curve'])
  fig.add_trace(go.Heatmap(x=x,y=y,z=CM,colorscale='Blues',text= CM.astype('str'), texttemplate="%{text}", showscale=False),row=1,col=1)

  # Plot Precision-Recall Curve
  y_onehot = pd.get_dummies(y_test, columns=[0,1])
  columns = ['Bad', 'Good']
  name = ['Bad', 'Good']
  for i in range(y_preds.shape[1]):
      y_true = y_onehot.iloc[:, i]
      y_pred = y_preds[:, i]
      precision, recall, _ = precision_recall_curve(y_true, y_pred)
      auc_score = average_precision_score(y_true, y_pred)
      fig.add_trace(go.Scatter(x=recall, y=precision, name=name[i], mode='lines'),row=1,col=2)

  fig.update_layout(height=500, width=1000)
  fig.show()

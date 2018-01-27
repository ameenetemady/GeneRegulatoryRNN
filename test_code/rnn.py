
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import os
os.system("pwd")
#training_set=pd.read_csv('BTCtrain.csv')   #reading csv file
training_set=pd.read_csv('./KO.tsv',delimiter="\t")
training_set2 = pd.read_table('./TFs.tsv',delimiter="\t")
#print(training_set.head())	
#print(training_set2.head()) 
#print(list(training_set2.columns.values))
#df = training_set.merge(training_set2, left_on='G9', right_on='G10', how='outer')
print("----------------------")
training_set1=training_set.iloc[:,1:7] 	 
training_set1=training_set1.values	  
				 
test_set = pd.read_table("./NonTFs.tsv",delimiter="\t")
test_set1 = test_set.iloc[:,1:7]
test_set1 = test_set1.values

sc = MinMaxScaler()		
training_set1 = sc.fit_transform(training_set1)
test_set1 = sc.fit_transform(test_set1)
xtrain=training_set1[0:3529]		  
ytrain=test_set1[0:3529]		  

inp_node=pd.DataFrame(xtrain[0:5])		   
out_node=pd.DataFrame(ytrain[0:5])         
ex= pd.concat([inp_node,out_node],axis=1)	  
#ex.columns=(['inp_node','out_node'])

# Reshaping into required shape for Keras
xtrain = np.reshape(xtrain, (3529, 7, 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


regressor=Sequential()			

regressor.add(LSTM(units=9,activation='sigmoid',input_shape=(None,1)))		

regressor.add(Dense(units=1))	

regressor.compile(optimizer='adam',loss='mean_squared_error') 		

regressor.fit(xtrain,ytrain,batch_size=32,epochs=90)	




# Reading CSV file into test set
test_set=pd.read_csv('../NonTFs.tsv',delimiter="\t")
test_set.head()


real_values = test_set.iloc[:,1:2]	

real_values = real_stock_price.values	

inputs = real_values

inputs = sc.transform(inputs)

inputs = np.reshape(inputs, (len(inputs), 1, 1))

predicted = regressor.predict(inputs)

predicted = sc.inverse_transform(predicted_stock_price)


#visualising the result

plt.plot(real_stock_price, color = 'red', label = 'Real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted',marker='--')
plt.title('RNN 7layer LSTM')
plt.legend()
plt.show()

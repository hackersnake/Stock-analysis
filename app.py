import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Define the start and end dates
start = '2010-01-01'
end = '2020-12-30'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter stock ticker')

try:
    df = yf.download(user_input, start=start, end=end)
    
    # Check if the DataFrame is empty
    if not df.empty:
        st.subheader('Data from 2010:')
        st.write(df.describe())  # Display the DataFrame
    else:
        st.write("No data available for the specified stock ticker and date range.")
except Exception as e:
    st.error(f"An error occurred: {e}")

st.subheader('Closing price vs Time chart')
fig=plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()  # Corrected: added ()
fig=plt.figure(figsize =(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()  # Corrected: added ()
ma200=df.Close.rolling(200).mean()  # Corrected: added ()
fig=plt.figure(figsize =(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)



X = df[['Close']]

X_train, X_test = train_test_split(X, test_size=0.3, shuffle=False)  # Set shuffle=False to preserve order

data_training = pd.DataFrame(X_train, columns=['Close'])
data_testing = pd.DataFrame(X_test, columns=['Close'])

print(data_training.shape)
print(data_testing.shape)


scaler=MinMaxScaler(feature_range = (0,1))

data_training_array=scaler.fit_transform(data_training)




model=load_model('kerass_model.h5')

past_100_days = data_training.tail(100)


final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_testl=[]
y_testl=[]

for i in range(100,input_data.shape[0]):
    x_testl.append(input_data[i-100:i])
    y_testl.append(input_data[i,0])

x_testl,y_testl=np.array(x_testl),np.array(y_testl)


x_testl_adjusted = x_testl[:, :50, :]

# Now, you can use the adjusted input data for prediction
y_pred = model.predict(x_testl_adjusted)

scaler=scaler.scale_

scale_factor = 1/scaler[0]
y_pred = y_pred * scale_factor
y_testl = y_testl * scale_factor

# Display the DataFrame and other plots

st.subheader('prediction vs original')
fig2 = plt.figure(figsize=(12, 6))  
plt.plot(y_testl, 'b', label='Original')
plt.plot(y_pred, 'r', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)  

# You might also want to display information about data shapes
st.write("Training data shape:", data_training.shape)
st.write("Testing data shape:", data_testing.shape)


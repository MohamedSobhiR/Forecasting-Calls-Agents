import joblib
import numpy as np 
import pandas as pd 

model = joblib.load('model01.h5')
scaler = joblib.load('scaler01.h5')

model2 = joblib.load('model02.h5')
scaler2 = joblib.load('scaler02.h5')


start_date = input("Start Prediction date : ")
end_date = input("End Prediction date : ")
start_offer_date = input("Start Offer date : ")
end_offer_date = input("End Offer date : ")
Offer_grad = int(input(("YOur Offer Grade : ")))
company = [1,0,0,0]

newdates = pd.Series(pd.date_range(start=start_date,end=end_date))
df2 = pd.DataFrame({'Display Date':newdates ,'Object Name_Company_B':company[0], 'Object Name_Company_C':company[1],
                    'Object Name_Company_D':company[2], 'Object Name_Company_E':company[3]})

offerdates = pd.Series(pd.date_range(start=start_offer_date,end=end_offer_date))
df3 = pd.DataFrame({'Display Date':offerdates , 'Offer_Grade' :Offer_grad })


df4 = pd.merge(df3, df2, how='right', on=['Display Date'])
df4 = df4.fillna(value = 0)

df4['Year'] = df4['Display Date'].dt.year
df4['Month'] = df4['Display Date'].dt.month
df4['Day'] = df4['Display Date'].dt.day
df4['Day_Name'] = df4['Display Date'].dt.day_name()
df4['Week_Day'] = df4['Display Date'].dt.weekday

df4.drop(['Display Date'] , axis = 1 ,  inplace = True )

df4 =df4[['Offer_Grade', 'Year', 'Month', 'Day', 'Week_Day',
       'Object Name_Company_B', 'Object Name_Company_C',
       'Object Name_Company_D', 'Object Name_Company_E']]

call_custom_data = scaler.transform(df4)

prediction = model.predict(call_custom_data)

print(f'the Calls count is {int(prediction.sum())}')




pre_input_with_Calls = {'Number of Calls T':int(prediction.sum()), 'Offer_Grade':Offer_grad, 'Year':int(end_date.split('-')[0]), 'Month':int(end_date.split('-')[1]) ,'Company_Company_B':company[0], 'Company_Company_C':company[1], 'Company_Company_D':company[2],'Company_Company_E':company[3]}

call_custom_data2 = np.array(list(pre_input_with_Calls.values()))

call_custom_data2 = scaler2.transform([call_custom_data2])

agent_prediction = model2.predict(call_custom_data2)


print(f'the Agents count is {int(agent_prediction)}')
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import datetime


''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
#print(dataset_2.to_string()) #This line will print out your data

##PROBELM 1

total = dataset_2['Brooklyn Bridge'] + dataset_2['Manhattan Bridge'] + dataset_2['Queensboro Bridge'] + dataset_2['Williamsburg Bridge']
dataset_2['Total'] = dataset_2['Brooklyn Bridge'] + dataset_2['Manhattan Bridge'] + dataset_2['Queensboro Bridge'] + dataset_2['Williamsburg Bridge']

#This code should find bike-brdige correlation for every bridge
correlation_BB = dataset_2['Brooklyn Bridge'].corr(dataset_2['Total'])
correlation_MB = dataset_2['Manhattan Bridge'].corr(dataset_2['Total'])
correlation_QB = dataset_2['Queensboro Bridge'].corr(dataset_2['Total'])
correlation_WB = dataset_2['Williamsburg Bridge'].corr(dataset_2['Total'])

#Find Bridge with highest correlation
corr_dict = {'Brooklyn Bridge': correlation_BB,'Manhattan Bridge': correlation_MB,'Queensboro Bridge': correlation_QB,'Williamsburg Bridge': correlation_WB}
        # I got this code by piecing togther variou infromation from online
sorted_corr = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1], reverse=True)}
best_bridges = list(sorted_corr)
print("The best bridges to install sensors on: ", best_bridges[:3])
print('Correlation of Brooklyn Bridge and Total: ', correlation_BB)
print('Correlation of Manhattan Bridge and Total: ', correlation_MB)
print('Correlation of Queensboro Bridge and Total: ', correlation_QB)
print('Correlation of Williamsburg Bridge and Total: ', correlation_WB)
''''
A dicitionary is created in descending order of correlation values. We want the three bridges with the highest 
correlation values which will end up being Manhattan, Queensboro, Williamsburg, Brooklyn has the lowest correlation
so we do not want a sensor there. We did this by calcuating the correlation between each bridges individual bike count and the total bike
count between the four bridges
'''


#Problem 2
print("\nLinear Model\n")
model = LinearRegression()

# Set up the independent variables and dependent variable
X = dataset_2[["Low Temp", "High Temp", "Precipitation"]]
y = dataset_2["Total"]

# Fit the model to the data
model.fit(X, y)

# Print the model coefficients
print("Intercept: ", model.intercept_)
print("Coefficients: ", model.coef_)
#Figure out how good the model fits the data
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("R-squared: ", r2)
print("\nEnd of Linear Model")



##This is a polynomial fit
print("\nBegin Polynomial Model(Quadratic)\n")
model = LinearRegression()

n = 0
while (n < 5):
        # Set up the independent variables and dependent variable
        Xp = dataset_2[["Low Temp", "High Temp", "Precipitation"]]
        yp = dataset_2["Total"]

        # Create polynomial features of degree 2
        poly = PolynomialFeatures(degree=n)
        X_poly = poly.fit_transform(Xp)

        # Fit the model to the transformed data
        model.fit(X_poly, yp)
        print("degree")
        print(n)
        # Print the model coefficients
        print("Intercept(polynomial): ", model.intercept_)
        print("Coefficients(polynomial): ", model.coef_)

        # Calculate the R-squared value
        yp_pred = model.predict(X_poly)
        r2p = r2_score(yp, yp_pred)
        print("R-squared(polynomia): ", r2p)

        print("\nEnd Polynomial fit\n")
        n = n + 1




#Problem 3
# Add a new column to the dataframe with the day of the week for each date
#dataset_2["DayOfWeek"] = pandas.to_datetime(dataset_2["Date"]).dt.day_name()
# Convert the "Date" column to a datetime object
#Plot a histogram
plt.figure(3)
plt.hist(dataset_2["Total"])
plt.xlabel("Total Bikers")
plt.ylabel("Frequency")
plt.title("Histogram of Total Number of Bikers")
plt.show()

dataset_2["Date"] = pandas.to_datetime(dataset_2["Date"], format="%d-%b")

# Extract the day of the week from the "Date" column and store it in a new "DayOfWeek" column
dataset_2["DayOfWeek"] = dataset_2["Date"].dt.day_name()


# Group the dataset by day of the week and calculate the average total bikers for each day
avg_traffic_by_day = dataset_2.groupby("Day").mean()["Total"] #Calc mean value of total
print(avg_traffic_by_day) #Shows average of total bikers per day

# Determine the day with the highest average total bikers
max_traffic_day = avg_traffic_by_day.idxmax() #This finds the day with the max amount of bikers
print("The day with the highest average total bikers is", max_traffic_day)













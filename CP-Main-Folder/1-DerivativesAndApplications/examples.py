# Derivatives and their applications. (4 points)
# 1. Give three examples of derivative’s application.
# 2. Select one example, use finite differences or compact finite differences, code and visualize


# Idea 1:
# Medical Imaging: Derivatives are used in medical imaging to create high-resolution images of
# the body's internal structures. For example, radiologists can use
# derivatives to enhance the contrast between different types of tissues,
# which can help them diagnose and treat a variety of medical conditions.

# Idea 2:
# Weather Forecasting: Derivatives are used in weather forecasting to predict
# changes in atmospheric conditions. By analyzing the derivatives of temperature,
# pressure, and other weather variables over time,
# meteorologists can forecast the likelihood of storms, hurricanes, and other weather events.

# Idea 3:
# Predicting stock prices: Using synthetic financial data, we can use
# derivatives to predict stock prices based on historical trends,
# news articles, and other market indicators.

# My choice: Idea 2 - Example of basic weather forecasting
import random
import numpy as np


def likelyhood(prevDay, currDay):
    '''
    This method calculates the likelyhood of rain,hurricane,sunny and cloudy days.
    Based on the finite difference of current and previous day parameters, we can guess 
    what the weather will look like tomorrow. Obviously it's a primitive representation of
    a very complicated procedure.
    '''
    pressureChange = currDay[0] - prevDay[0]
    humidityChange = currDay[1] - prevDay[1]
    temperatureChange = currDay[2] - prevDay[2]
    if pressureChange > 10 and humidityChange > 10 and temperatureChange < -5 and currDay[0] >= 80:
        return "Hurricane"
    elif temperatureChange < -2 and humidityChange > 10 and currDay[1] >= 50:
        return "Rain"
    elif currDay[2] >= 50:
        return "Sunny"
    else:
        return "Cloudy"


def generate_weather_data(num):
    '''
    Generates synthetic data, dayList contains 'real' parameters of any given day,
    meanwhile prognoseList contains assumed or forecasted weather words
    '''
    dayList = [] * num
    prognoseList = [] * (num - 1)
    for i in range(num):
        pressure = int(random.random() * 100)
        humidity = int(random.random() * 100)
        temperature = int(random.random() * 100)
        dayList.append((pressure, humidity, temperature))

    for i in range(1, num):
        prognoseList.append(likelyhood(dayList[i - 1], dayList[i]))

    return (prognoseList,dayList)


prog,day = generate_weather_data(100)
for i in range(len(prog)):
    print("Assumption:",prog[i],"-- Real parameters:",day[i])

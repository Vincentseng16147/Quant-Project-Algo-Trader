import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# These indicators are for consumer staple stocks, due to their defensive nature. We seek to capture the 
# divergence from indexes that the stock is a part of, hypothesizing a signal to predict its movement


#######            STOCK LIST:           #############
#Proctor & Gamble: PG
pg = yf.Ticker("PG").history(period="5y")

#Coca-cola: KO
ko = yf.Ticker("KO").history(period="5y")

#Walmart: WMT
wmt = yf.Ticker("WMT").history(period="5y")

#Costco: COST
cost = yf.Ticker("COST").history(period="5y")

#Altria Group: MO
mo = yf.Ticker("MO").history(period="5y")

#Colgate-Palmolive:CL
cl = yf.Ticker("CL").history(period="5y")

#######        PROXY INDEX LIST:           #############

#Dow Jones US Consumer Staple Index: ^DJUSST
djcons= yf.Ticker("^DJUSST").history(period="5y")

#S&P500 Consumer Staples: ^SP500-30
spcons = yf.Ticker("^SP500-30").history(period="5y")

############################################################################################################################

#Setting up main dataframe

maindf = pd.concat(
    {"PG": pg["Close"],"KO": ko["Close"],"WMT": wmt["Close"], 
    "COST": cost["Close"], "MO": mo["Close"], "CL": cl["Close"],
    "Dow Jones": djcons["Close"], "S&P": spcons["Close"]}, axis = 1).sort_index()

retsDF = maindf.pct_change()

volumedf = pd.concat(
    {"PG": pg["Volume"],"KO": ko["Volume"],"WMT": wmt["Volume"], 
    "COST": cost["Volume"], "MO": mo["Volume"], "CL": cl["Volume"]}, axis = 1).sort_index()


############################################################################################################################

#Indicator 1: momentum. 2 dataframes of 20 day moving averages subtracted by the index for both indicies.


maDF = retsDF.rolling(20).mean()

excessDJDF = pd.DataFrame({
            "PG": maDF["PG"] - maDF["Dow Jones"],
            "KO": maDF["KO"] - maDF["Dow Jones"],
            "WMT": maDF["WMT"] - maDF["Dow Jones"],
            "COST": maDF["COST"] - maDF["Dow Jones"],
            "MO": maDF["MO"] - maDF["Dow Jones"],
            "CL": maDF["CL"] - maDF["Dow Jones"],
        })
excessDJDF = excessDJDF.add_suffix("_maD_vs_DJ")
excessDJDF = excessDJDF*100

excessSPDF = pd.DataFrame({
            "PG": maDF["PG"] - maDF["S&P"],
            "KO": maDF["KO"] - maDF["S&P"],
            "WMT": maDF["WMT"] - maDF["S&P"],
            "COST": maDF["COST"] - maDF["S&P"],
            "MO": maDF["MO"] - maDF["S&P"],
            "CL": maDF["CL"] - maDF["S&P"],
        })
excessSPDF = excessSPDF.add_suffix("_maD_vs_S&P500")
excessSPDF = excessSPDF*100


print(excessDJDF.tail())
print(excessSPDF.tail())

#So, to recap our output is a DF of a 20 day moving average of RETURNS, for each stock subtracted by the 20 day moving average returns of the DJ index and SP500 index respectively
#for use you may have to clear out the first few rows since they are NA due to moving average. Everthing is in percentage form.
#these numbers are PERCENTAGES, negative means it underpreforms, positive means it overpreforms, so the inverse is the signal

######################################################################################################################

#Indicator 2: Volatility

retVol20 = retsDF.rolling(20).std()

relVolDJDF = pd.DataFrame({
            "PG": retVol20["PG"] / retVol20["Dow Jones"],
            "KO": retVol20["KO"] / retVol20["Dow Jones"],
            "WMT": retVol20["WMT"] / retVol20["Dow Jones"],
            "COST": retVol20["COST"] / retVol20["Dow Jones"],
            "MO": retVol20["MO"] / retVol20["Dow Jones"],
            "CL": retVol20["CL"] / retVol20["Dow Jones"],
        })
relVolDJDF = relVolDJDF.add_suffix("_reVol_vs_DJ")
relVolDJDF = (relVolDJDF - 1)*100

relVolSPDF = pd.DataFrame({
            "PG": retVol20["PG"] / retVol20["S&P"],
            "KO": retVol20["KO"] / retVol20["S&P"],
            "WMT": retVol20["WMT"] / retVol20["S&P"],
            "COST": retVol20["COST"] / retVol20["S&P"],
            "MO": retVol20["MO"] / retVol20["S&P"],
            "CL": retVol20["CL"] / retVol20["S&P"],
        })
relVolSPDF = relVolSPDF.add_suffix("_reVol_vs_S&P500")
relVolSPDF = (relVolSPDF - 1)*100

print(relVolDJDF.tail())
print(relVolSPDF.tail())

#this measures relative volatility (SD) compared to each index. This is in PERCENTAGES

#I think idea should be more risk = more reward, but we can discuss what to do with this number, its mainly going to be used for future indicators I'd imagine

###############################################################################################################################################################

#Indicator 3, Z score: Stock mean - Index mean over Stock SD for 20dayMA. Idea is the index mean is the true mean and we use stock mean and deviation to see how it deviates

basicZDJDF = pd.DataFrame({
        "PG" : ((maDF["PG"] - maDF["Dow Jones"])/retVol20["PG"]),
        "KO" : ((maDF["KO"] - maDF["Dow Jones"])/retVol20["KO"]),
        "WMT" : ((maDF["WMT"] - maDF["Dow Jones"])/retVol20["WMT"]),
        "COST" : ((maDF["COST"] - maDF["Dow Jones"])/retVol20["COST"]),
        "MO" : ((maDF["MO"] - maDF["Dow Jones"])/retVol20["MO"]),
        "CL" : ((maDF["CL"] - maDF["Dow Jones"])/retVol20["CL"]),
    })
basicZSPDF = pd.DataFrame({
        "PG" : ((maDF["PG"] - maDF["S&P"])/retVol20["PG"]),
        "KO" : ((maDF["KO"] - maDF["S&P"])/retVol20["KO"]),
        "WMT" : ((maDF["WMT"] - maDF["S&P"])/retVol20["WMT"]),
        "COST" : ((maDF["COST"] - maDF["S&P"])/retVol20["COST"]),
        "MO" : ((maDF["MO"] - maDF["S&P"])/retVol20["MO"]),
        "CL" : ((maDF["CL"] - maDF["S&P"])/retVol20["CL"]),
    })
print(basicZDJDF.tail())
print(basicZSPDF.tail())

#basicZDJDF[["PG","KO","WMT"]].plot(figsize=(11,5))
#plt.axhline(0, color="black", linestyle="--", linewidth=1)
#plt.title("Relative Z-Scores vs Dow")
#plt.ylabel("Z-Score")
#plt.show()

# this is pretty cool and actually works, feel free to plot it I included it at the end. 
#It's a z score, so the higher the more far off it is and needs a correction, negative means underpreforming positive is overpreforming the index

################################################################################################################################################################

#Indicator 4, Volume Z-Score - Take the rolling z score of the volume

volumeMa = volumedf.rolling(20).mean()
volumeSd = volumedf.rolling(20).std()

volumeZ = pd.DataFrame({
    "PG" : ((volumedf["PG"] - volumeMa["PG"])/volumeSd["PG"]),
    "KO" : ((volumedf["KO"] - volumeMa["KO"])/volumeSd["KO"]),
    "WMT" : ((volumedf["WMT"] - volumeMa["WMT"])/volumeSd["WMT"]),
    "COST" : ((volumedf["COST"] - volumeMa["COST"])/volumeSd["COST"]),
    "MO" : ((volumedf["MO"] - volumeMa["MO"])/volumeSd["MO"]),
    "CL" : ((volumedf["CL"] - volumeMa["CL"])/volumeSd["CL"]),
})
print("this is volume indicator")
print(volumeZ.tail())
volumeZ[["PG","KO","WMT"]].plot(figsize=(11,5))
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Volume Z scores")
plt.ylabel("Z-Score")
plt.show()
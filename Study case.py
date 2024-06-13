import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
df = pd.read_csv('developer_dataset.csv')

#print(df.describe())
#The mean of a work week in hours for a programmer is 41 hours
#We had 111209 repplies in the survey
#The mean years of coding of people who replied the survey is 9 years and half, ranging from 0 years of experience and 50

#Columns dropped because of 60% or more missing data
df.drop(["NEWJobHunt","NEWJobHuntResearch", "NEWLearn"],
        axis=1,
        inplace=True)

df[['RespondentID','Country']].groupby('Country').count()

missingData = df[['Employment','DevType']].isnull().groupby(df['Country']).sum().reset_index()

A=sns.catplot(
    data=missingData, kind="bar",
    x="Country", y="Employment",
    height = 6, aspect = 2,
    palette='Blues')
A.set_axis_labels("Country","Missing Employment Data")

B=sns.catplot(
    data=missingData, kind="bar",
    x="Country", y="DevType",
    height = 6, aspect = 2,
    palette="Greens")

B.set_axis_labels("Country","Missing DevType Data")
plt.show()
plt.clf()
#No Irregular missind data type

# Dropping rows where 'Employment' or 'DevType' are NaN
df.dropna(subset=['Employment','DevType'],
          inplace=True,
          how='any')
# Plotting count of respondents by 'Country' and 'Employment'
empfig = sns.catplot(x="Country", col="Employment",
                     data=df, kind='count',
                     height=6, aspect = 1.5)
empfig.set_axis_labels("Country","Count")
empfig.set_titles("{col_name}")

# Creating a new DataFrame for 'Country' and 'DevType'
devdf = df[["Country", "DevType"]].copy()
# Initializing new columns to False
devdf['BackEnd'] = devdf['DevType'].str.contains('back-end', case=False, na=False)
devdf['FrontEnd'] = devdf['DevType'].str.contains('front-end', case=False, na=False)
devdf['FullStack'] = devdf['DevType'].str.contains('full-stack', case=False, na=False)
devdf['Mobile'] = devdf['DevType'].str.contains('mobile', case=False, na=False)
devdf['Admin'] = devdf['DevType'].str.contains('administrator', case=False, na=False)

devdf = devdf.melt(id_vars=["Country"],
                    value_vars=["BackEnd","FrontEnd","FullStack","Mobile","Admin"],
                    var_name='DevCat',
                    value_name='DevFlag')

devdf = devdf[devdf["DevFlag"]]


devFig = sns.catplot(x="Country", col="DevCat",
                     data=devdf, kind='count',
                     height=6, aspect=1.5)
plt.show()
plt.clf()

missingUndergrad = df["UndergradMajor"].isnull().groupby(df["Year"]).sum().reset_index()

sns.catplot(x="Year", y="UndergradMajor",
            data=missingUndergrad, kind="bar",
            height=4, aspect=1)

df = df.sort_values(["RespondentID","Year"])
df["UndergradMajor"].bfill(axis=0, inplace=True)

# Filtering relevant columns and dropping NaN values
edudf = df[["Year", "UndergradMajor"]].copy()
edudf.dropna(how='any', inplace=True)

# Creating new boolean columns based on 'UndergradMajor'
edudf['SocialScience'] = edudf['UndergradMajor'].str.contains('(?i)social science', case=False, na=False)
edudf['NaturalScience'] = edudf['UndergradMajor'].str.contains('(?i)natural science', case=False, na=False)
edudf['ComSci'] = edudf['UndergradMajor'].str.contains('(?i)computer science', case=False, na=False)
edudf['Development'] = edudf['UndergradMajor'].str.contains('(?i)development', case=False, na=False)
edudf['OtherEng'] = edudf['UndergradMajor'].str.contains('(?i)another engineering', case=False, na=False)
edudf['NoMajor'] = edudf['UndergradMajor'].str.contains('(?i)never declared', case=False, na=False)

# Melting the DataFrame for plotting
edudf = edudf.melt(id_vars=['Year'],
                   value_vars=['SocialScience', 'NaturalScience', 'ComSci', 'Development', 'OtherEng', 'NoMajor'],
                   var_name='EduCat',
                   value_name='EduFlag')

# Filtering rows where 'EduFlag' is True
edudf = edudf[edudf['EduFlag']]

# Grouping and counting the number of occurrences
edudf = edudf.groupby(['Year', 'EduCat']).size().reset_index(name='EduFlag')

# Plotting the data
eduFig = sns.catplot(x='Year', y='EduFlag', col='EduCat',
                     data=edudf, kind='bar',
                     height=6, aspect=1.5)
eduFig.set_axis_labels("Year", "Count")
eduFig.set_titles("{col_name}")

# Showing the plot
plt.show()
plt.clf()

# Plotting YearsCodePro and ConvertedComp by Year
compFields = df[["Year", "YearsCodePro", "ConvertedComp"]]

# Plot for YearsCodePro
D = sns.boxplot(x="Year", y="YearsCodePro", data=compFields)
plt.title("Years of Professional Coding Experience by Year")
plt.show()
plt.clf()

# Plot for ConvertedComp
E = sns.boxplot(x="Year", y="ConvertedComp", data=compFields)
plt.title("Converted Compensation by Year")
plt.show()
plt.clf()

imputedf = df[["YearsCodePro", "ConvertedComp"]]

traindf, testdf = train_test_split(imputedf, train_size=0.1, random_state=0)

imp = IterativeImputer(max_iter=20, random_state=0)
imp.fit(imputedf)

compdf = pd.DataFrame(np.round(imp.transform(imputedf), 0), columns=["YearsCodePro", "ConvertedComp"])

compPlotdf = compdf[compdf['ConvertedComp'] <= 150000]

compPlotdf['CodeYearBins'] = pd.qcut(compPlotdf['YearsCodePro'], q=5)

sns.boxplot(x="CodeYearBins", y="ConvertedComp", data=compPlotdf)
plt.title("Converted Compensation by Years of Professional Coding Experience (Binned)")
plt.show()
plt.clf()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

#Load Dataset
url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
df = pd.read_csv(url)

#Inspect
print(df.info())
print(df.describe())

#Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode)

#Remove duplicates
df = df.drop_duplicates()

#Filter data: Passengers in first class
first_class = df[df["Pclass"] == 1]
print("First Class Passengers: \n", first_class.head())

# Bar Chart: Survival rate by class
survival_by_class = df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar", color="green")
plt.title("Survival Rate by Class")
plt.ylabel("Survival Rate")
plt.show()

#Histogram: Age distribution
sns.histplot(df["Age"], kde=True, bins=20, color="skyblue")
plt.title("Age Distribution")
plt.ylabel("Age")
plt.ylabel("Frequency")
plt.show()

#Scatter Plot: Age vs Fare
plt.scatter(df["Age"], df["Fare"], alpha=0.5, color="purple")
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

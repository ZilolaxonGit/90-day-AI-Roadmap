import pandas as pd
import numpy as np

data = {
    "name" : ["Ali", "Marjona", "Zilola"],
    "grades" : [0, 1, None]
}

df =pd.DataFrame(data)

# Counting the number of missing values
print(df.isna().sum())

# Dropping the missing values
print(df.dropna())

# If all values are missing
df2 = df.dropna(how="all")


print(df2)

# df["score"] = df["score"].fillna()

print(df)

df.drop_duplicates(inplace=True)

print(df)

diagram = """

    PANDAS CLEANING
│
├── Missing Values
│   ├── isna()
│   ├── dropna()
│   └── fillna()
│
├── Duplicates
│   └── drop_duplicates()
│
├── Text Cleaning
│   ├── strip()
│   ├── lower()
│   ├── title()
│   └── replace()
│
├── Filtering
│   ├── remove negative values
│   ├── keep ranges
│   └── multiple conditions
│
└── Mini Project
    ├── load raw CSV
    ├── clean text
    ├── fill missing
    ├── remove duplicates
    └── save clean CSV


"""

data = {
    "name" : ["Zilola ", " Iroda", "Malika", "Muhammadjon", "Muhammadjon"],
    "age" : [None, 34, 12, 56, 56],
    "is_active" : [True, False, None, False, False]
}

df = pd.DataFrame(data)

df["name"] = df["name"].str.strip()
df["age"] = df["age"].fillna(df["age"].mean())
df.drop_duplicates(inplace=True)
df.to_csv("cleaned_students_list.csv", index=False)
print(df)

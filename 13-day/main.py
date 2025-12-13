import pandas as pd

data = {
    "name": ["Ali", "Sara", "John"],
    "age": [17, 18, 19],
    "score": [90, 88, 75]
}

df = pd.DataFrame(data)

print(df)
print(df.info())
print(df[["name", "age"]])
print(df[df["score"] > 85])

tasks = """
Task 1

Create a DataFrame of 5 students with columns:

name

age

math_score

english_score

"""

# Task 1

data = {
    "name" : ["Ali", "Marjona", "Ruxshona", 'Saida', "Maftuna"],
    "age" : [10, 45, 23, 34, 12],
    "math_score" : [10, 30, 40, 20, 42],
    "english_score" : [67, 78, 34, 45, 23]
}

df = pd.DataFrame(data)
df["averge_score"] = [12, 23, 45, 12, 45]
print(df)

df = pd.read_csv("13-day/data.csv")
print(df)

print(df.head())
print(df.info())
print(df.describe())
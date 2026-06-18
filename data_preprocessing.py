import pandas as pd
import os

print("🚀 Starting preprocessing...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("✅ Files loaded successfully!")
print("Fake shape:", fake.shape)
print("True shape:", true.shape)

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], axis=0)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data.to_csv("data/combined.csv", index=False)

print("✅ combined.csv created successfully!")
print("Final shape:", data.shape)
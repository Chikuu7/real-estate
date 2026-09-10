
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


file_path = "House Price India.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully ✅")
print(df.head())


# Remove duplicates
df = df.drop_duplicates()

# Drop null values
df = df.dropna()

print("\nData cleaned ✅")

features = [
    'living area',
    'lot area',
    'number of bedrooms',
    'number of bathrooms',
    'number of floors',
    'waterfront present',
    'number of views',
    'Built Year'
]

X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Done ✅")


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully ✅")


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE")
print("MAE:", mae)
print("R2 Score:", r2)


importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\n🔥 Feature Importance:")
print(feature_importance_df)

# Plot
plt.figure()
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()


output_df = X_test.copy()
output_df['Actual Price'] = y_test
output_df['Predicted Price'] = y_pred

output_df.to_csv("predictions_output.csv", index=False)

print("\nPredictions saved as predictions_output.csv ✅")



sample_house = np.array([[2000, 3000, 3, 2, 2, 1, 4, 2015]])

predicted_price = model.predict(sample_house)

print("\n🏠 Sample House Prediction:")
print("Predicted Price:", predicted_price[0])

# ================================
# 11. SAVE FULL VISUAL OUTPUT IMAGE
# ================================

plt.figure(figsize=(12, 8))

# Plot 1: Actual vs Predicted
plt.subplot(2, 1, 1)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")

# Plot 2: Feature Importance
plt.subplot(2, 1, 2)
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()

# Save image
plt.savefig("model_output.png")

print("\n📸 Image saved as model_output.png ✅")
#Samyak
#chikuu
#Chikuu1
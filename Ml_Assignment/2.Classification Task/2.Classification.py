import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ta

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,roc_auc_score, confusion_matrix, roc_curve, auc, classification_report

# Loading Dataset
df = pd.read_csv(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\2.Classification Task\woo-BTC USDT.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[df['timestamp'].dt.year != 2017]
df['year'] = df['timestamp'].dt.year
df = df.set_index('timestamp')

# Feature Engineering
df['returns'] = df['close'].pct_change().round(2)
df['target'] = df['close'].shift(-1)
df['sma'] = df['close'].shift(1).rolling(window=7).mean().round(2)
df['ema'] = df['close'].shift(1).ewm(span=7, adjust=False).mean().round(2)

bb = ta.volatility.BollingerBands(close=df['close'].shift(1), window=7)
df['bbh'] = bb.bollinger_hband().round(2)
df['bbm'] = bb.bollinger_mavg().round(2)
df['bbl'] = bb.bollinger_lband().round(2)

df['volatility'] = df['close'].shift(1).rolling(window=7).std().round(2)
df['rsi'] = ta.momentum.RSIIndicator(close=df['close'].shift(1), window=14).rsi().round(2)
df['close_mom_7'] = (df['close'].shift(1) / df['close'].shift(8) - 1).round(2)

df.dropna(inplace=True)

# Target Label for Classification
df['next_day_close'] = df['close'].shift(-1)
df['target'] = (df['next_day_close'] > df['close']).astype(int)
df.dropna(inplace=True)

# Drop Unused Columns
df = df.drop(columns=['returns', 'next_day_close'])
final_df = df.drop(columns=['close'])

train_df = final_df[final_df['year'].between(2018, 2022)].drop(columns=['year'])
test_df = final_df[final_df['year'] == 2023].drop(columns=['year'])
forecast_df = final_df[final_df['year'] >= 2024].drop(columns=['year'])

X_train, y_train = train_df.drop(columns=['target']), train_df['target']
X_test, y_test = test_df.drop(columns=['target']), test_df['target']
X_forecast, y_forecast = forecast_df.drop(columns=['target']), forecast_df['target']

# pipeline
pipeline = joblib.load(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\2.Classification Task\clf_lr_pipeline.pkl')

y_pred_lr = pipeline.predict(X_forecast)
y_prob_lr = pipeline.predict_proba(X_forecast)[:, 1]

# def function to evaluate the model
def evaluate_forecast(model, X, y_true, model_name="Model"):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"Evaluation Report - {model_name} on Forecast Data")
    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:\n", report)

    # Class-wise metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).T.loc[['0', '1'], ['precision', 'recall', 'f1-score']]
    metrics_df.plot(kind='bar', figsize=(6, 4), title=f"{model_name} - Class-wise Metrics")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title(f'{model_name} - Confusion Matrix')
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")

    ax[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].set_title(f'{model_name} - ROC Curve')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

evaluate_forecast(pipeline, X_forecast, y_forecast, model_name="Logistic Regression")
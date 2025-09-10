{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "61d1e1ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8c0b7ba9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_and_save_model(data_path):\n",
    "    \n",
    "    df = pd.read_csv(data_path)\n",
    "    df['Date'] = pd.to_datetime(df['Date'])\n",
    "    categorical_features = ['Region', 'Country']\n",
    "    df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True)\n",
    "\n",
    "    train_df = df_processed[df_processed['Dataset Split'] == 'train']\n",
    "    \n",
    "    TARGET = 'Deforestation Event (Yes=1,No=0)'\n",
    "    features_to_drop = [\n",
    "        'Tile ID', 'Image ID', 'Date', 'Dataset Split', 'Image File Path',\n",
    "        'Predicted Risk Score', 'Biome Type', TARGET\n",
    "    ]\n",
    "    \n",
    "    X_train = train_df.drop(columns=features_to_drop, errors='ignore')\n",
    "    y_train = train_df[TARGET]\n",
    "\n",
    "    scaler = StandardScaler()\n",
    "    X_train_scaled = scaler.fit_transform(X_train)\n",
    "\n",
    "    model = LogisticRegression(random_state=42, max_iter=1000)\n",
    "    model.fit(X_train_scaled, y_train)\n",
    "\n",
    "    joblib.dump(model, 'logistic_model.joblib')\n",
    "    joblib.dump(scaler, 'scaler.joblib')\n",
    "    joblib.dump(list(X_train.columns), 'model_columns.joblib')\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    train_and_save_model('deforestation.csv')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

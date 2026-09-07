
#Importing libraries and setting the size parameter of sns plot

import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import Lasso, LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier


sns.set(rc={'figure.figsize':(11.7,8.27)})

def data_preprocessing(train_df, X_test):
    #Here we can observer that some columns have standard deviation nearly 0.
    #They are incapable of contributing in the performance of model, so we will drop them later.
    drop1=list(filter(lambda keys: train_df[keys].value_counts().max()> 0.99*train_df.shape[0],train_df.keys()))
    #creating the list of the columns that have quasi constant features as i will not drop them, will just segregate
    #as we can see there are no quasi constant features.

    #As in my dataset the number of duplicated rows is already 0 so i have not written any code for dropping these values.


    #Test Train Splitting

    #X_train y_train splitting and label encoding
    X_train = train_df.drop(['Target'], axis =1)
    target_mapping = {'low': 0, 'medium': 1, 'high': 2}
    y_train = train_df['Target'].map(target_mapping)

    return X_train,y_train

def EDA(X_train,y_train,X_test):
    #Removing Rows of those columns that have NULL values less than 2 percent

    # Step 1: Calculate the 2% threshold of total rows
    threshold = 0.02 * X_train.shape[0]

    # Step 2: Loop through columns and remove rows where NaN count is less than 2% of total rows
    for column in X_train.columns:
        # Count NaNs in the column
        nan_count = X_train[column].isna().sum()

        # If the NaN count in the column is less than the 2% threshold, remove the rows
        if nan_count < threshold:
            # Identify rows to keep
            rows_to_keep = ~X_train[column].isna()

            # Filter both X_train and y_train based on the rows to keep
            X_train = X_train[rows_to_keep]
            y_train = y_train[rows_to_keep]

    #Removing columns that have more than 90% null values.

    # Calculate the percentage of missing values for each column
    missing_percentage = (X_train.isnull().sum() / len(X_train)) * 100

    # Identify columns with more than 90% missing values
    columns_to_drop = missing_percentage[missing_percentage > 90].index

    dropped_columns=list(columns_to_drop.copy())

    #Imputing Values using Correlation Matrix that have compartively less NULL values

    # Impute missing 'HarvestProcessingType' values in X_train
    X_train['HarvestProcessingType'] = X_train['HarvestProcessingType'].fillna(
        X_train.groupby('RawLocationId')['HarvestProcessingType'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
    )
    X_train['HarvestProcessingType'] = X_train['HarvestProcessingType'].fillna(
        X_train.groupby('NationalRegionCode')['HarvestProcessingType'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
    )
    X_train['HarvestProcessingType'] = X_train['HarvestProcessingType'].fillna(
        X_train.groupby('DistrictId')['HarvestProcessingType'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
    )

    # Fallback to overall mode if still missing
    overall_mode_harvest = X_train['HarvestProcessingType'].mode().iloc[0]
    X_train['HarvestProcessingType'] = X_train['HarvestProcessingType'].fillna(overall_mode_harvest)

    # Now, apply the same filling strategy to X_test based on X_train values
    # Map the mode values from X_train to X_test using RawLocationId, NationalRegionCode, and DistrictId
    X_test['HarvestProcessingType'] = X_test['HarvestProcessingType'].fillna(
        X_test['RawLocationId'].map(X_train.groupby('RawLocationId')['HarvestProcessingType'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ))
    )
    X_test['HarvestProcessingType'] = X_test['HarvestProcessingType'].fillna(
        X_test['NationalRegionCode'].map(X_train.groupby('NationalRegionCode')['HarvestProcessingType'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ))
    )
    X_test['HarvestProcessingType'] = X_test['HarvestProcessingType'].fillna(
        X_test['DistrictId'].map(X_train.groupby('DistrictId')['HarvestProcessingType'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ))
    )

    # Fallback to overall mode from X_train if still missing
    X_test['HarvestProcessingType'] = X_test['HarvestProcessingType'].fillna(overall_mode_harvest)



    # Impute missing 'CultivatedAreaSqft1' values in X_train
    X_train['CultivatedAreaSqft1'] = X_train['CultivatedAreaSqft1'].fillna(
        X_train.groupby('TotalCultivatedAreaSqft')['CultivatedAreaSqft1'].transform('mean')
    )
    X_train['CultivatedAreaSqft1'] = X_train['CultivatedAreaSqft1'].fillna(
        X_train.groupby('MainIrrigationSystemCount')['CultivatedAreaSqft1'].transform('mean')
    )
    X_train['CultivatedAreaSqft1'] = X_train['CultivatedAreaSqft1'].fillna(
        X_train.groupby('TaxAgrarianValue')['CultivatedAreaSqft1'].transform('mean')
    )

    # Fallback to the whole column mean if still missing
    X_train['CultivatedAreaSqft1'] = X_train['CultivatedAreaSqft1'].fillna(X_train['CultivatedAreaSqft1'].mean())

    # Now, apply the same filling strategy to X_test based on X_train values

    # Step 1: Fill missing values in 'CultivatedAreaSqft1' in X_test based on 'TotalCultivatedAreaSqft' using X_train logic
    X_test['CultivatedAreaSqft1'] = X_test['CultivatedAreaSqft1'].fillna(
        X_test['TotalCultivatedAreaSqft'].map(X_train.groupby('TotalCultivatedAreaSqft')['CultivatedAreaSqft1'].transform('mean'))
    )

    # Step 2: Fill remaining missing values in 'CultivatedAreaSqft1' based on 'MainIrrigationSystemCount'
    X_test['CultivatedAreaSqft1'] = X_test['CultivatedAreaSqft1'].fillna(
        X_test['MainIrrigationSystemCount'].map(X_train.groupby('MainIrrigationSystemCount')['CultivatedAreaSqft1'].transform('mean'))
    )

    # Step 3: Fill remaining missing values in 'CultivatedAreaSqft1' based on 'TaxAgrarianValue'
    X_test['CultivatedAreaSqft1'] = X_test['CultivatedAreaSqft1'].fillna(
        X_test['TaxAgrarianValue'].map(X_train.groupby('TaxAgrarianValue')['CultivatedAreaSqft1'].transform('mean'))
    )

    # Fallback - Fill any remaining missing values in 'CultivatedAreaSqft1' with the whole column mean from X_train
    X_test['CultivatedAreaSqft1'] = X_test['CultivatedAreaSqft1'].fillna(X_train['CultivatedAreaSqft1'].mean())


    # Impute missing 'FieldSizeSqft' values in X_train
    X_train['FieldSizeSqft'] = X_train['FieldSizeSqft'].fillna(
        X_train.groupby('TownId')['FieldSizeSqft'].transform('mean')
    )

    # Fallback to the whole column mean if still missing
    X_train['FieldSizeSqft'] = X_train['FieldSizeSqft'].fillna(X_train['FieldSizeSqft'].mean())

    # Now, apply the same filling strategy to X_test based on X_train values

    # Step 1: Fill missing values in 'FieldSizeSqft' in X_test based on 'TownId' using X_train logic
    X_test['FieldSizeSqft'] = X_test['FieldSizeSqft'].fillna(
        X_test['TownId'].map(X_train.groupby('TownId')['FieldSizeSqft'].transform('mean'))
    )

    # Fallback - Fill any remaining missing values in 'FieldSizeSqft' with the whole column mean from X_train
    X_test['FieldSizeSqft'] = X_test['FieldSizeSqft'].fillna(X_train['FieldSizeSqft'].mean())


    # Impute missing 'SoilFertilityType' values in X_train using the mode
    mode_soil_fertility = X_train['SoilFertilityType'].mode()[0]
    X_train['SoilFertilityType'] = X_train['SoilFertilityType'].fillna(mode_soil_fertility)

    # Now, apply the same imputation to X_test using the mode from X_train
    X_test['SoilFertilityType'] = X_test['SoilFertilityType'].fillna(mode_soil_fertility)

    return dropped_columns,X_train,y_train,X_test

def histogramAnalysis(dropped_columns,X_train,y_train,X_test):
      return dropped_columns,X_train,y_train,X_test

"""#Feature Engineering"""

def enhance_features(data):
    print("\nEnhancing Features...")

    # Tax burden ratio and agrarian value ratio
    data['TaxBurden'] = data['TotalTaxAssessed'] / (data['TotalValue'] + 1)
    data['AgrarianValueProportion'] = data['TaxAgrarianValue'] / (data['TotalValue'] + 1)

    # Farm age based on field establishment year
    data['FarmAgeYears'] = data['ValuationYear'].max() - data['FieldEstablishedYear']

    # Irrigation count considering partial systems
    data['TotalIrrigationSystems'] = (data["MainIrrigationSystemCount"] +
                                      (data["PartialIrrigationSystemCount"].fillna(0) * 0.5))

    # Irrigation density per cultivated area
    data['IrrigationAreaDensity'] = data['TotalIrrigationSystems'] / data['TotalCultivatedAreaSqft']

    # Water resource availability by combining access points and reservoirs
    data['WaterResources'] = data['WaterAccessPoints'] + data['WaterReservoirCount'] + \
                             (data['NaturalLakePresence'] * 0.5) * (data['WaterAccessPoints'] + data['WaterReservoirCount'])

    # Water resource density per cultivated area
    data['WaterAreaDensity'] = data['WaterResources'] / data['TotalCultivatedAreaSqft']

    # Field efficiency considering various factors
    data['FieldUtilization'] = (data['SoilFertilityType'] + data['WaterResources'] +
                                data['FarmVehicleCount'] + data['TotalIrrigationSystems'] +
                                data['StorageAndFacilityCount']) / data['FieldSizeSqft']

    return data

def onehotencoding_and_imputing(dropped_columns,X_train,y_train,X_test):
      X_train = pd.get_dummies(X_train, columns=['LandUsageType'], prefix='LandUsageType')
      dropped_columns.append('LandUsageType')

      X_train = pd.get_dummies(X_train, columns=['HarvestProcessingType'], prefix='HarvestProcessingType')
      dropped_columns.append('HarvestProcessingType')

      X_train = pd.get_dummies(X_train, columns=['TypeOfIrrigationSystem'], prefix='TypeOfIrrigationSystem')
      dropped_columns.append('TypeOfIrrigationSystem')

      X_train = pd.get_dummies(X_train, columns=['SoilFertilityType'], prefix='SoilFertilityType')
      dropped_columns.append('SoilFertilityType')

      #Apply the same one-hot encoding to X_test
      X_test = pd.get_dummies(X_test, columns=['TypeOfIrrigationSystem'], prefix='TypeOfIrrigationSystem')
      X_test = pd.get_dummies(X_test, columns=['HarvestProcessingType'], prefix='HarvestProcessingType')
      X_test = pd.get_dummies(X_test, columns=['LandUsageType'], prefix='LandUsageType')
      X_test = pd.get_dummies(X_test, columns=['SoilFertilityType'], prefix='SoilFertilityType')

      #Align the columns of X_test with X_train by adding missing columns (with all zeros)
      X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

      # Identify columns with missing values in X_train and X_test
      columns_with_nulls_train = X_train.columns[X_train.isna().any()]
      columns_with_nulls_test = X_test.columns[X_test.isna().any()]

      # Perform mean imputation using only the means from X_train
      for col in columns_with_nulls_train:
          # Impute missing values in X_train with its mean
          mean_value = X_train[col].mean()
          X_train[col] = X_train[col].fillna(mean_value)

          # Impute missing values in X_test with the same mean from X_train
          if col in X_test.columns:
              X_test[col] = X_test[col].fillna(mean_value)

      # For any columns in X_test with missing values that aren't in X_train,
      # fill with a constant (e.g., 0 or another placeholder value)
      for col in columns_with_nulls_test:
          if col not in columns_with_nulls_train:
              X_test[col] = X_test[col].fillna(0)  # or choose another placeholder value


      dropped_columns.append('Latitude')
      dropped_columns.append('Longitude')
      dropped_columns.append('UID')

      return dropped_columns,X_train,y_train,X_test

def identify_highly_correlated_columns(dropped_columns,data, threshold=0.9):
    # Step 1: Compute the correlation matrix
    corr_matrix = data.corr().abs()

    # Step 2: Create an upper triangle matrix mask to ignore duplicate pairs
    upper_tri = corr_matrix.mask(np.tril(np.ones(corr_matrix.shape)).astype(bool))

    # Step 3: Identify pairs of columns with correlation above the threshold
    columns_to_drop = set()
    for col in upper_tri.columns:
        high_corr_cols = upper_tri.index[upper_tri[col] > threshold].tolist()
        for high_corr_col in high_corr_cols:
            # Choose the column with more null values to drop
            if data[col].isnull().sum() <= data[high_corr_col].isnull().sum():
                columns_to_drop.add(high_corr_col)
            else:
                columns_to_drop.add(col)

    # Update the dropped_columns list with identified columns to drop
    dropped_columns.extend(columns_to_drop)

    return dropped_columns

def select_features(X_train, X_test, y_train, n_features=25):
    """
    Select features using multiple methods and voting.
    Returns selected train and test features.
    """
    feature_scores = {feature: 0 for feature in X_train.columns}

    # 1. Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    for feature, importance in zip(X_train.columns, rf.feature_importances_):
        feature_scores[feature] += importance

    # 2. Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
    mi_selector.fit(X_train, y_train)
    for feature, score in zip(X_train.columns, mi_selector.scores_):
        feature_scores[feature] += score / np.max(mi_selector.scores_)

    # 3. Lasso Feature Selection
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    for feature, coef in zip(X_train.columns, np.abs(lasso.coef_)):
        feature_scores[feature] += coef / np.max(np.abs(lasso.coef_))

    # Select top features
    selected_features = pd.DataFrame({
        'Feature': feature_scores.keys(),
        'Score': feature_scores.values()
    }).nlargest(n_features, 'Score')['Feature'].tolist()

    return X_train[selected_features], X_test[selected_features]

def pca_and_model_training(X_train_selected, X_test_selected, y_train):
      # PCA was evaluated during development but is not used in the final model.

      # Define the Balanced Random Forest Classifier model
      model = BalancedRandomForestClassifier(
          n_estimators=100,
          sampling_strategy='all',  # Change default from 'auto' to 'all'
          replacement=True,         # Change default from False to True
          bootstrap=False,          # Change default from True to False
          random_state=42
      )


      # Fit the model on the training data
      model.fit(X_train_selected, y_train)

      # Make predictions on the test data
      y_pred = model.predict(X_test_selected)

      return y_pred

import argparse
def make_predictions(test_fname, predictions_fname):
    #loading the train dataset
    df=pd.read_csv('../data/train.csv')
    train_df=df.copy()

    #loading the test dataset
    X_test=pd.read_csv(test_fname)

    X_train,y_train=data_preprocessing(train_df, X_test)

    dropped_columns,X_train,y_train,X_test=EDA(X_train,y_train,X_test)

    dropped_columns,X_train,y_train,X_test=histogramAnalysis(dropped_columns,X_train,y_train,X_test)

    X_train=enhance_features(X_train)
    X_test=enhance_features(X_test)

    dropped_columns,X_train,y_train,X_test= onehotencoding_and_imputing(dropped_columns,X_train,y_train,X_test)

    dropped_columns=identify_highly_correlated_columns(dropped_columns,X_train[list(X_train.columns.difference(dropped_columns))])

    # Prepare X_train_features and X_test_features by excluding the dropped columns
    X_train_features = X_train[list(X_train.columns.difference(dropped_columns))]
    X_test_features = X_test[list(X_train.columns.difference(dropped_columns))]

    X_train_selected, X_test_selected = select_features(X_train_features, X_test_features, y_train)


    y_pred=pca_and_model_training(X_train_selected, X_test_selected, y_train)

    target_mapping = {0:'low', 1:'medium', 2:'high'}

    # Remap y_pred using the mapping
    y_pred_mapped = pd.Series(y_pred).map(target_mapping)

    result_df = pd.DataFrame({
    'UID': X_test['UID'],
    'Target': y_pred_mapped
    })

      # Write the submission data to a CSV file
    result_df.to_csv(predictions_fname, index=False)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, help='file path of train.csv')
    parser.add_argument("--test-file", type=str, help='file path of test.csv')
    parser.add_argument("--predictions-file", type=str, help='save path of predictions')
    args = parser.parse_args()
    make_predictions(args.test_file, args.predictions_file)

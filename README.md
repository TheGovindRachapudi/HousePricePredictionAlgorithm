# HousePricePredictionAlgorithm

# Real Estate Price Estimator

## Overview
The **Real Estate Price Estimator** is a software tool designed to predict the price categories of properties (affordable, mid-range, and premium housing) based on various property attributes. The prediction is powered by Random Forest Classification and Multiple Regression Analysis, leveraging the **Zillow House Value Dataset** from Kaggle. 

The tool employs efficient data structures like **Heaps** for filtering properties, querying price ranges, and clustering similar properties. It also offers a range of data visualizations to help analyze trends and feature correlations.

## Features
### Core Functionality:
- **Price Prediction**: Predicts property price categories (affordable, mid-range, premium).
- **Efficient Queries**: Quickly retrieves cheapest and most expensive properties using heaps.
- **Data Visualization**: Displays price distributions, feature correlations, and predicted vs. actual price trends.

### Supported Data Structures:
- **Heaps**: Used for querying cheapest and most expensive properties.
- **Lists**: Stores property details for sequential feature processing.
- **Dictionaries**: Provides fast access to property attributes.
- **DataFrames**: Organizes tabular data from the dataset.

## Dataset
This tool utilizes the **Zillow House Value Dataset** from Kaggle. Ensure you have the dataset in CSV format to load and preprocess.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/real-estate-price-estimator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd real-estate-price-estimator
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```plaintext
.
├── data
│   └── house_data.csv         # Dataset (ensure this file exists in the `data` folder)
├── src
│   ├── dataloader.py          # DataLoader class
│   ├── feature_extract.py     # FeatureExtract class
│   ├── manual_heap.py         # ManualHeap class
│   ├── property_management.py # PropertyManagement class
│   ├── price_predictor.py     # PricePredictor class
│   └── visuals.py             # Visualization utilities
├── tests
│   └── test_cases.py          # Test cases for each module
├── README.md                  # Project description
└── requirements.txt           # Required Python libraries
```

## Key Classes and Methods

### 1. **DataLoader**
- **Methods**:
  - `load_data(filepath)`: Loads dataset into a DataFrame.
  - `clean_data(raw_data)`: Cleans missing/invalid values from the dataset.

### 2. **FeatureExtract**
- **Methods**:
  - `extract_features(dataframe)`: Normalizes features like location and size.
  - `encode_categorical(column)`: Converts categorical features to numerical values.

### 3. **ManualHeap**
- **Methods**:
  - `compare(parent, child)`: Determines heap property (min or max heap).
  - `push(key, data)`: Inserts an element into the heap.
  - `pop()`: Removes the root element and maintains heap properties.

### 4. **PropertyManagement**
- **Methods**:
  - `add_properties()`: Extracts attributes like price, beds, baths, state, and city.
  - `get_cheap_properties(n)`: Retrieves the n cheapest properties.
  - `get_expensive_properties(n)`: Retrieves the n most expensive properties.

### 5. **PricePredictor**
- **Methods**:
  - `train_model(training_data)`: Trains the Random Forest Classification model.
  - `predict_price(new_data)`: Predicts price categories for new properties.
  - `plot_accuracy(features, target, max_trees, step)`: Visualizes model training accuracy.

### 6. **Visuals**
- **Methods**:
  - `plot_price_distribution(dataframe)`: Visualizes price distributions.
  - `plot_feature_correlations(dataframe)`: Displays feature correlations.
  - `plot_predicted_vs_actual(results, actual_col, predicted_col)`: Compares predicted vs. actual price categories.
  - `plot_price_distribution_by_category(dataframe, price_column, category_column)`: Shows price distributions by category.

## Algorithms Used
- **Heap Sort**: Efficiently sorts property prices for querying.
- **Random Forest Classification**: Predicts price categories using robust ensemble learning.

## Libraries
- **Pandas**: For data analysis and manipulation.
- **Scikit-learn**: Implements regression, classification, and model evaluation.
- **Matplotlib**: Data visualization.
- **Seaborn**: Simplifies Matplotlib visualizations.
- **NumPy**: Array processing and linear algebra operations.

## Test Cases
| Test Case | Description                                                                                      | Input                                   | Output                                                                                       |
|-----------|--------------------------------------------------------------------------------------------------|-----------------------------------------|----------------------------------------------------------------------------------------------|
| TC1       | Validates data loading and cleaning.                                                             | Raw dataset                            | Cleaned data with minimal missing values and a new Price Category column                    |
| TC2       | Verifies feature extraction and encoding.                                                        | Cleaned data                           | DataFrame with scaled numerical features and encoded categorical features                   |
| TC3       | Ensures proper heap operations (insertions, deletions).                                          | Property attributes                     | Min heap retrieves smallest key, max heap retrieves largest key                             |
| TC4       | Tests Random Forest model training and predictions.                                              | Processed features and target labels   | Model accuracy score, predicted price categories                                            |
| TC5       | Checks accurate retrieval of cheapest and most expensive properties.                             | Cleaned property data                  | DataFrames with top 5 cheapest and most expensive properties                                |

## Usage
1. Load and preprocess the dataset:
   ```python
   from src.dataloader import DataLoader
   data_loader = DataLoader()
   cleaned_data = data_loader.clean_data(data_loader.load_data('data/house_data.csv'))
   ```
2. Extract features:
   ```python
   from src.feature_extract import FeatureExtract
   feature_extractor = FeatureExtract()
   processed_data = feature_extractor.extract_features(cleaned_data)
   ```
3. Train the model and make predictions:
   ```python
   from src.price_predictor import PricePredictor
   predictor = PricePredictor()
   predictor.train_model(training_data)
   predictions = predictor.predict_price(new_data)
   ```
4. Visualize results:
   ```python
   from src.visuals import Visuals
   visuals = Visuals()
   visuals.plot_price_distribution(cleaned_data)
   ```

## Contributions
Contributions are welcome! Please fork the repository and create a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

Enjoy predicting real estate prices efficiently!

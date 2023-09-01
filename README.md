# Crypto Trading Volume Analysis

This repository contains a Python script that I have created to analyze the trading volume data of a cryptocurrency for my personal use. 

## Description

The script `main.py` performs the following steps:

1. Load the trading volume data from a CSV file. The data should have at least two columns: a timestamp (`snapped_at`) and the corresponding trading volume (`volume`). 
2. Plot the volume over time.
3. Identify and remove outliers using the Interquartile Range (IQR) method.
4. Plot the cleaned volume data over time.
5. Analyze and visualize the average trading volume by day of the week, month, and whether it's a weekend or not.

## Usage

The script can be used as follows:

```python
from main import AdvancedDataAnalysisPipeline

pipeline = AdvancedDataAnalysisPipeline("path_to_your_data.csv")

```
## Note
This script was specifically designed for my own trading volume data and as per my specific requirements. The methods and processes used may not be universally applicable or accurate for all types of data. Please use this as a reference and adjust the code according to your needs.

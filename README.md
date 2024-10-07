# Stock-Forecasting-Model

This Python script uses a vanilla Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical closing price data. It allows users to select from a predefined list of stock symbols and visualize both historical data and price predictions plotted against actual prices. 

## Features

- Select from multiple stock symbols in the command line interface (SPY, META, AMZN, AAPL, NFLX, GOOG)
- Visualize historical stock prices and trading volume
- Train a LSTM model to predict future stock prices
- Evaluate model performance using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
- Visualize predictions against actual stock prices

## Requirements

- Python 3.x
- pandas
- matplotlib
- numpy
- tensorflow
- scikit-learn

## Installation

1. Clone this repository or download the script.
2. Install the required libraries:

```bash
pip install pandas matplotlib numpy tensorflow scikit-learn
```

3. Ensure you have the CSV files for each stock symbol (SPY.csv, META.csv, AMZN.csv, AAPL.csv, NFLX.csv, GOOG.csv) in the same directory as the script.

## Usage

1. Run the script:

```bash
python stocksproject.py
```

2. Follow the command line prompts to select a stock symbol and specify the number of training epochs.
3. The script will display visualizations. Close both windows to continue the program.
4. The script will then begin training, displaying runtime metrics. When it is done, the final prediction will be visualized. You can use the built-in matplotlib interface to zoom to rectangle to see just how accurate the model was. The command line will also disply MSE and RMSE for specific error metrics. 

## Code Structure

- **Dataset selection**: User selects a stock symbol, and the corresponding CSV file is loaded.
- **Data visualization**: Historical stock prices and trading volume are plotted.
- **Data preparation**: The closing price data is scaled and prepared for training.
- **LSTM model**: A sequential LSTM model is defined and trained on the prepared data.
- **Prediction and evaluation**: The model predicts stock prices for the test set, and performance metrics are calculated.
- **Results visualization**: Actual vs. predicted stock prices are plotted for comparison.

## Customization

- Modify the list of stock symbols and corresponding CSV files to include additional stocks.
- Adjust the LSTM model architecture by changing the number of units, layers, or adding dropout for potentially improved performance.
- Experiment with the number of epochs (you can do this through the command line!)

## Bugs
- **Data visualization**: Integrating the volume data with the stock price in matplotlib causes an extra empty window to appear in the initial visualization. I still haven't figured out if this is an issue with my code or with the library itself, but it's something to be aware of nonetheless.


## Notes

- This script uses a simple LSTM model and does not capture all factors affecting stock prices. In fact, this model is currently more of a line predictor than a stock predictor, since this would work the same way with any other bivariable dataset. Eventually a random forest and/or support vector machine will be implemented to include the correlation of stock volume and price in the final prediction. In the long run, natural language processing may be leveraged to further enhance the model by parsing financial articles. 

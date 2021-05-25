# Algorithmic Trading with Deep Learning

## Summary
This notebook trains a deep learning model to predict the direction a trading bot should take in the current time step so as to achieve positive return in next time step.  In other words, the model predicts whether the asset return is positive or negative in the next time step and take action to long or short position based on analytics known up to the current point in time.

### Trading Instrument
Currency pair **USDJPY** is used here.  Bid ask spread of this instrument is used in estimating the transaction cost.  USDJPY has the lowest spread when compared with other currency pairs.

### Data
Using Oanda API for historical data download and executing trades.

### History
90 days of latest historical prices.  The data is split with a ratio of 4:1 to training data and out-of-sample data.

### Time step
15-minute time step.  This means the algorithm will make a decision to take a long or short position.  Transaction cost is only taken in consideration when there is a switch in position's direction.

### Notebook

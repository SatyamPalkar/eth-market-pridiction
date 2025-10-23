# Data Dictionary - Ethereum Historical Data

## Dataset Overview
- **Source:** CoinMarketCap
- **Cryptocurrency:** Ethereum (ETH)
- **Date Range:** 2015-01-01 to 2025-01-01
- **Frequency:** Daily
- **Total Records:** ~3,650 days

## Column Descriptions

### Timestamp Columns

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `timeOpen` | ISO 8601 Timestamp | Start timestamp of the daily interval (00:00:00 UTC) |
| `timeClose` | ISO 8601 Timestamp | End timestamp of the daily interval (23:59:59 UTC) |
| `timeHigh` | ISO 8601 Timestamp | Timestamp when the highest price occurred during the day |
| `timeLow` | ISO 8601 Timestamp | Timestamp when the lowest price occurred during the day |
| `timestamp` | ISO 8601 Timestamp | Reference timestamp for the conversion currency value |

### Identifier Column

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `name` | Integer | CoinMarketCap cryptocurrency ID (2781 for Ethereum) |

### Price Columns (OHLC)

| Column Name | Data Type | Description | Unit |
|-------------|-----------|-------------|------|
| `open` | Float | Opening price at the start of the day | USD |
| `high` | Float | Highest price during the day | USD |
| `low` | Float | Lowest price during the day | USD |
| `close` | Float | Closing price at the end of the day | USD |

### Volume & Market Cap

| Column Name | Data Type | Description | Unit |
|-------------|-----------|-------------|------|
| `volume` | Float | Total trading volume (adjusted) for the day | USD |
| `marketCap` | Float | Market capitalization by circulating supply | USD |

## Target Variable

For this project, we create an additional column:

| Column Name | Data Type | Description | Calculation |
|-------------|-----------|-------------|-------------|
| `target_high_next_day` | Float | Next day's HIGH price (prediction target) | `df['high'].shift(-1)` |

## Data Quality Notes

1. **Missing Values:** Check for any gaps in the date sequence
2. **Outliers:** Look for unusual spikes in price or volume
3. **Data Consistency:** Verify that `low ≤ open, close ≤ high`
4. **Volume Anomalies:** Very low volume days may indicate data quality issues

## Example Row

```python
timeOpen:    2024-12-31T00:00:00.000Z
timeClose:   2024-12-31T23:59:59.999Z
timeHigh:    2024-12-31T14:35:00.000Z
timeLow:     2024-12-31T01:32:00.000Z
name:        2781
open:        92643.25
high:        96090.60
low:         91914.03
close:       93429.20
volume:      43625106842.87
marketCap:   1850183342119.26
timestamp:   2024-12-31T23:59:59.999Z
```

## Feature Engineering Ideas

### Derived Features

1. **Price-based:**
   - Daily return: `(close - open) / open`
   - High-low spread: `high - low`
   - Price range: `(high - low) / low`

2. **Lag features:**
   - Previous day prices: `lag_1_close`, `lag_2_close`, etc.
   - Previous day returns

3. **Rolling statistics:**
   - Moving averages: 7-day, 14-day, 30-day
   - Rolling volatility: Standard deviation
   - Rolling min/max

4. **Technical indicators:**
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - EMA (Exponential Moving Average)

5. **Volume-based:**
   - Volume change
   - Volume moving average
   - Price-volume trend

6. **Time-based:**
   - Day of week
   - Month
   - Quarter
   - Is weekend
   - Day of year

## Statistical Summary

Run this to get summary statistics:
```python
df.describe()
```

Expected ranges (approximate):
- **Price (2015-2025):** $0.50 - $4,800
- **Volume:** Highly variable, billions in USD
- **Market Cap:** Billions to hundreds of billions USD

## Data Preprocessing Checklist

- [ ] Convert timestamp columns to datetime
- [ ] Sort data chronologically
- [ ] Check for missing values
- [ ] Identify and handle outliers
- [ ] Verify data consistency (low ≤ close ≤ high, etc.)
- [ ] Create target variable
- [ ] Engineer features
- [ ] Split data chronologically (train/test)
- [ ] Scale features if needed

## References

- [CoinMarketCap API Documentation](https://coinmarketcap.com/api/documentation/)
- [Ethereum Historical Data](https://coinmarketcap.com/currencies/ethereum/)


# Immo-Eliza: Predicting Belgian House Prices

Can we predict the price of a house or apartment in Belgium from its features? This is the second step of the Immo-Eliza project at BeCode:

1. [Scraping](https://github.com/muntadherabdulhusein/immo-eliza-scraping) property listings
2. **Machine learning model** (this repo)
3. [Deployment](https://github.com/muntadherabdulhusein/immo-eliza-deployment) of the model as an API

## The data

`properties.csv` has about 75,000 Belgian properties with 30 columns: price, type (house, apartment, villa…), region, province, locality, living area, number of bedrooms, garden, swimming pool and more.

## What I did

- **Cleaning:** dropped latitude/longitude and ID, filled missing numbers with the mean and removed rows where important text values were missing. About 20,000 complete properties were left.
- **Encoding and scaling:** one-hot encoded the categorical columns (type, region, locality…) and standardised the numbers.
- **Exploring:** looked at which features are most related to the price.
- **Models:** started with a simple linear regression on one feature, then multiple linear regression on the top 10 features, and finally **XGBoost**.

## Results

| Model | R² | RMSE |
|---|---|---|
| XGBoost | **0.82** | about €208,000 |

The XGBoost model explains about 82% of the variation in prices. The RMSE is still quite high, mostly because of very expensive properties (villas, exceptional properties) that are hard to predict.

The features most related to the price:

- living area (`total_area_sqm`)
- number of bedrooms
- being in Brugge
- having a swimming pool
- being a villa

## Files

- `immo.ipynb`: exploration, cleaning and all the models
- `main.py`: the cleaning and XGBoost training as a script
- `xgboost_model.pkl`: the trained model, used in the deployment repo
- `properties.csv`: the dataset

## Run it

```
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python main.py
```

## What I would improve

- Remove extreme outliers before training, or predict the log of the price
- Tune the XGBoost parameters with cross-validation
- Use the latitude/longitude instead of dropping them


# write a pseudocode for the following code

```python
lst = []
for i in range(100):
    for tick in returns.columns:
        day = np.random.randint(0, len(returns) - 5)
        samp = returns.iloc[day: day+5, returns.columns.get_loc(tick)]
        lst.append(list(samp))
```

```
1. initialize an empty list
2. for i = 1, 2, 3, ..., 100:
3.     for each ticker in the S&P 500:
4.         randomly select 5 consecutive days from the returns data
5.         append the 5-day returns to the list
6. return the list
```

pseudocode of a random hyperparameter search
```
d = dict where keys are hyperparameters and values are
    lists of possible values

1. for i = 1, 2, 3, ..., number of models:
2.     initialize an empty dictionary
3.     for each key in d:
4.         randomly select a value from the list 
           of possible values and store it in the
           dictionary
5.     train a model with the selected hyperparameters
6.     generate 1000 samples from the model
```
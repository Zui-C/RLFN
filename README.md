## Training

Check the config file ```config.yaml``` 

```
python train.py
```

## Testing

```
python inference.py --dataset 2014
```


## best_thin.pth
Not save MathBERTa's params



## Practical Application Notes

If you aim to use this model for real-world applications, it's recommended to train on larger datasets. Finding data that matches your specific application scenario can greatly enhance the model's performance. Also, make sure to update vocabulary-related files, including `'words_dict.txt'` and `'token.json'`.

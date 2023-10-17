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


## GPT-4V test
[GPT-4V test](https://github.com/Zui-C/RLFN/edit/main/GPT-4V_test) on CROHME 2014 datasets done by JiaQi Han


## Practical Application Notes

If you aim to use this model for real-world applications, it's recommended to train on larger datasets. Finding data that matches your specific application scenario can greatly enhance the model's performance. Also, make sure to update vocabulary-related files, including `'words_dict.txt'` and `'token.json'`.


## Acknowledgments

We would like to acknowledge the work done on [SAN](https://github.com/tal-tech/SAN) and its modified version [CAN](https://github.com/LBH1024/CAN). Additionally, we have utilized [MathBERTa](https://github.com/witiko/scm-at-arqmath3) in our work. These work have been valuable references for our project.



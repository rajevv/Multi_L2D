
# Hatespeech Experiments
## Obtain data
1. Although it's already available in this filder, you can download `labeled_data.csv` from original repository:
```
wget https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv
```

1. Process the data with `process_text_data.py`:
```
python3 process_text_data.py
```

The output will be:
- `labeled_data.csv`
- `data.pkl`: We will work with this file. 
- `data_vectorized.pkl`
- `input_full.pkl`
- `tweets.txt`
  
For more information please check the original repository from Nastaran Okati et al.: (https://github.com/Networks-Learning/differentiable-learning-under-triage/tree/main/Hatespeech)

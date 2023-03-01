

## Obtain data
1. Download `labeled_data.csv` from original repository:
```
wget https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv
```

2. Process the data with `process_text_data.py`:
```
python3 process_text_data.py
```

3. 
## Code structure

<!--  - `train.ipynb` Trains our method and the baselines and generates all the figures illustrated in the paper.
 - `process_text_data.py` Loads the hatespeech tweets from `labeled_data.csv` and extracts 100 dimensional features using `fasttext` and saves human predictive model based on the human annotations. Please refer to section 6 of our paper for more details. If you want to generate the data use the following command:  `python3 generate_text_data.py`
 - `/models` Contains the trained model of our method and the baselines.
 -->
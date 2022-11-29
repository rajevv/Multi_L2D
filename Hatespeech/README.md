## Code structure

 - `train.ipynb` Trains our method and the baselines and generates all the figures illustrated in the paper.
 - `process_text_data.py` Loads the hatespeech tweets from `labeled_data.csv` and extracts 100 dimensional features using `fasttext` and saves human predictive model based on the human annotations. Please refer to section 6 of our paper for more details. If you want to generate the data use the following command:  `python3 generate_text_data.py`
 - `/models` Contains the trained model of our method and the baselines.

# Disaster Response Pipeline Project

### Project 
The objective of the project is to build a model for an API that classifies disaster messages. To implement this model data skills of engeneering and software engineering were necessary. Creating a machine learning pipeline to categorize these events alows  to send the messages to an appropriate disaster relief agency.

### Datasets
#### Data set containing real messages that were sent during disaster events. This disaster data were provided by Appen (formally Figure 8) 

- "disaster_categories" provided the 36 categories of classification of the messages

- "disaster_messages" provided the 26216 messages to fit the ML model 

### Libraries Necessary:
- nlk, sklearn, pandas, numpy, re, sqlalchemy, pickle and sys

### Steps:

- process_data.py: implementation of ETL process
- DisasterResponse.db: database stored at the end of ETL process
- train_classifier.py: build the Machine Learn Model that classify the messages into categories
- run.py: initiate the web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

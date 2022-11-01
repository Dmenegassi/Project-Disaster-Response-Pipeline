# Disaster Response Pipeline Project

### Project Description
To analyze disaster data from Appen (formally Figure 8) and build a model for an API that classifies disaster messages. That is an application of how data engeneering and  software engeneering skils can be applyied to solve real world problems with efficiency

## Files and Descriptions

- img1: pre visualization from web app
- img2: pre visualization from web app


### app
- Template
- go.html : classification result page of web app
- master.html : Main web page
- run.py : Flask file to run the web app

### data
- DisasterResponse.db: database saved after ETL process
- disaster_categories.csv : data
- disaster_messages.csv : data
- process_data.py: run the ETL process 

### models
- classifier.pkl: saved model
- train_classifier.py: Machine Lern process

### README.md

### Python version and Libraries

- Python3
- Pandas
- Numpy
- NLK
- Scikit
- SQLalchemy
- Pickle
- Flask
- Plotly


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Acknowledgements
- Udacity for the Data Science Nanodegree Program
- Figure Eight for providing the data

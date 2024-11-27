# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - Set your cmd in proyect directory `cd your/path`
    - Install requirements.txt `python -m pip install -r requirements.txt`
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db random_forest_tuned.pkl multi_label.pkl` 
    If you need to execute ML algorithm with new pkl files, change names for .pkl files

2. Run the following to run your web app.
    `python app/run.py`

3. Go to http://127.0.0.1:3001/
![Screenshot](static/images/screenshot_app_1.png)
![Screenshot 2](static/images/screenshot_app_2.png)
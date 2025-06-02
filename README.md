# TextVerify - A better AI Detection Model
This project is for teachers or other people within the academic sphere that wish to test writing to see if it is AI generated. The project takes in an input text and a prompt and will tell the user whether the text was AI generated or not. 

## Abstract 
This study is in the AI Detection field. It aims to create a better AI Detector that will be used to detect AI in students' writing. The topic of AI Detection is very important because it is crucial to make sure that students are submitting their own work and upholding integrity in the learning environment. Also, current AI Detectors don’t work very well, so there is room for improvement in this field. The gap in current research involves the AI detection accuracy, with many detectors not being accurate in detecting AI. Some of these websites even have disclaimers that warn against using the AI Detection as an end all be all, as they aren’t reliable. Our research aim was to make an AI Detector that would reliably detect AI content. The main method to do this involved Retrieval Methods, Doc2Vec encoding, and Cosine Similarity Scores.

## Making sense of the code
App.py is the main program that is being run for this project, that is where all the work is done. The Templates folder is for HTML templates that are displayed on the website. Reset_db.py and Reset_db.sql are used to initialize the sql database. Database.db is the file that stores the actual database.

## Installation
1. Clone the repo:  `git clone https://github.com/devsb123/AI-Detector-using-Retrieval-Methods`
2. Navigate to the directory: `cd AI-Detector-using-Retrieval-Methods`
3. Install dependencies: `npm install`
## Usage
1. Create and export an API Key (generate an API key and then export it as an environmental variable in your terminal)
   https://platform.openai.com/docs/libraries?desktop-os=windows&language=python use this website for help
3. Install the SDK using pip: Type in pip install openai
4. Run the following commands in your terminal:
   ```bash
   python reset_db.py
   python app.py
5. This will start a local web server.
6. Open your web browser and go to:
 http://127.0.0.1/
7. Follow the instructions displayed on the website to proceed.
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

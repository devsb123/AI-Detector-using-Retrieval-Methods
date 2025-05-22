#import the required libraries
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import math
from openai import OpenAI
client = OpenAI()

app = Flask(__name__)

#initialize important variables that will be used throughout the program
currentStudentGivenText = ""
lstOfAllDocuments = []
promptToGenerateResponsesOn = ""
minimumWordCount = 200
maximumWordCount = 500
threshold = 0.60
aiGenerated = False
entryThatMostCloselyMatchesInputText = ""
aiGeneratedAsAString = ""
highestCosineSimilarityScore = 0

#connect to the SQL database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def clear_table():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM textDatabase")
    conn.commit()
    conn.close()

#make a route for the first screen that the user sees
@app.route('/')
def initialWebpage():
    return render_template('startingTemplate.html')

#this is a second route for after the user enters in intial information
@app.route('/results', methods=['POST'])
def routeForResults():
    if request.method == 'POST':
        #gather the important variables values
        global promptToGenerateResponsesOn
        promptToGenerateResponsesOn = request.form.get("textEntered")
        global minimumWordCount
        minimumWordCount = int(request.form.get("minWordCount"))
        global maximumWordCount
        maximumWordCount = int(request.form.get("maxWordCount"))
        global numberOfResponsesToGenerate
        numberOfResponsesToGenerate = 20

        global currentStudentGivenText
        global lstOfAllDocuments

        currentStudentGivenText = request.form.get("inputText")

        global totalPromptToGiveGPT
        totalPromptToGiveGPT = promptToGenerateResponsesOn + f"Give a response between {minimumWordCount} and {maximumWordCount} words."

        #use GPT's API to get text to fill up the database
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": totalPromptToGiveGPT
            }],
        n = numberOfResponsesToGenerate)

        lstOfAllDocuments = [response.message.content for response in completion.choices] + [currentStudentGivenText]

        #run the Doc2Vec model to get vector representation of the documents 

        #preprocess the documents, and create TaggedDocuments
        tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                    tags=[str(i)]) for i,
                    doc in enumerate(lstOfAllDocuments)]
        
        #train the Doc2vec model
        model = Doc2Vec(vector_size=25,
                        min_count=2, epochs=80)
        model.build_vocab(tagged_data)
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        
        #get the document vectors
        document_vectors = [model.infer_vector(
            word_tokenize(doc.lower())) for doc in lstOfAllDocuments]
        listFormatForDocVectors = [vector.tolist() for vector in document_vectors]

        clear_table()

        #insert the vectors and the documents into the database
        conn = get_db_connection()
        for i, vectorArray in enumerate(listFormatForDocVectors[0:len(listFormatForDocVectors)-1]):
            conn.execute("INSERT INTO textDatabase (itemNum, textEntered, encodedVectorEntry1, encodedVectorEntry2, encodedVectorEntry3, encodedVectorEntry4, encodedVectorEntry5, encodedVectorEntry6, encodedVectorEntry7, encodedVectorEntry8, encodedVectorEntry9, encodedVectorEntry10, encodedVectorEntry11, encodedVectorEntry12, encodedVectorEntry13, encodedVectorEntry14, encodedVectorEntry15, encodedVectorEntry16, encodedVectorEntry17, encodedVectorEntry18, encodedVectorEntry19, encodedVectorEntry20, encodedVectorEntry21, encodedVectorEntry22, encodedVectorEntry23, encodedVectorEntry24, encodedVectorEntry25) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ((i+1, lstOfAllDocuments[i], vectorArray[0], vectorArray[1], vectorArray[2], vectorArray[3], vectorArray[4], vectorArray[5], vectorArray[6], vectorArray[7], vectorArray[8], vectorArray[9], vectorArray[10], vectorArray[11], vectorArray[12], vectorArray[13], vectorArray[14], vectorArray[15], vectorArray[16], vectorArray[17], vectorArray[18], vectorArray[19], vectorArray[20], vectorArray[21], vectorArray[22], vectorArray[23], vectorArray[24])))
               
        conn.commit()
        data = conn.execute('SELECT * FROM textDatabase').fetchall()
        conn.close()
        
        database_encoded_vectors = []
        for row in data:
            encoded_vector = row[2:27]
            database_encoded_vectors.append(list(encoded_vector))
        
        currentVectorEncodedText = document_vectors[-1]
        studentTextVectorAsAList = currentVectorEncodedText.tolist()

        #Do calculations involving Cosine Similarity to find the closest entry in the database

        squaredCurrentVectorEncodedText = [i*i for i in studentTextVectorAsAList]   
        sumOfSquaredCurrentVectorEncodedText = sum(squaredCurrentVectorEncodedText)
        magnitudeOfInputVector = math.sqrt(sumOfSquaredCurrentVectorEncodedText)


        cosineSimilarityScores = []
        
        for paragraph in database_encoded_vectors:
            dotProduct =  sum([studentTextVectorAsAList[index] * value for index, value in enumerate(paragraph)])
            magnitudeOfParagraph = math.sqrt(sum([i * i for i in paragraph]))
            cosineSimilarityScores.append(dotProduct / (magnitudeOfParagraph * magnitudeOfInputVector))

        

        global highestCosineSimilarityScore
        highestCosineSimilarityScore = max(cosineSimilarityScores)
        indexOfHighestCosineSimilarityScore = cosineSimilarityScores.index(highestCosineSimilarityScore)
        global entryThatMostCloselyMatchesInputText
        entryThatMostCloselyMatchesInputText = lstOfAllDocuments[indexOfHighestCosineSimilarityScore]
        global threshold
        global aiGenerated
        if(highestCosineSimilarityScore >= threshold):
            aiGenerated = True
        global aiGeneratedAsAString
        aiGeneratedAsAString = "not AI generated"
        if(aiGenerated == True):
            aiGeneratedAsAString = "AI Generated"
        #Second Screen, confirms what user had typed in
        return render_template('secondScreen.html', prompt = promptToGenerateResponsesOn, minWords = minimumWordCount, maxWords = maximumWordCount)

#render a template for whether it was AI Generated or not, using Cosine Similarity Score and Threshold
@app.route('/aiGenOrNot')
def routeForAIGenOrNot():
    return render_template("finalScreen.html", closestEntry = entryThatMostCloselyMatchesInputText, AIGEN = aiGeneratedAsAString, thres = threshold, cosSimilarity = highestCosineSimilarityScore, inputText = currentStudentGivenText)

app.debug = True
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
#webpage 
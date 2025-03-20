from flask import Flask, render_template, request, redirect, url_for
from gensim.models import Word2Vec
import sqlite3
import numpy as np
import json
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import math

app = Flask(__name__)

itemNum = 0
currentText = ""
lstOfText = []
currentVectorEncodedText = []
lstOfVectorEncodedText = []
promptToGenerateResponsesOn = ""
minimumWordCount = 200
maximumWordCount = 500
threshold = 0.95
aiGenerated = False
entryThatMostCloselyMatchesInputText = ""
aiGeneratedAsAString = ""
highestCosineSimilarityScore = 0

def decode_if_bytes(data):
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    return data

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def initialWebpage():
    return render_template('startingTemplate.html')

#@app.route('/')
#def index():
    #return render_template('index.html')

"""@app.route('/results', methods=['POST'])
def routeForResults():
    if request.method == 'POST':
        global itemNum 
        global currentText
        global lstOfText
        global currentVectorEncodedText
        global lstOfVectorEncodedText

        itemNum += 1
        currentText = request.form.get("textEntered")
        lstOfText.append(currentText)

        #if(lstOfText == []):
           # currentVectorEncodedText = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
           # lstOfVectorEncodedText = []

        #preprocess the documents, and create TaggedDocuments
        tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                    tags=[str(i)]) for i,
                    doc in enumerate(lstOfText)]
        
        #train the Doc2vec model
        model = Doc2Vec(vector_size=25,
                        min_count=2, epochs=80)
        model.build_vocab(tagged_data)
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        
        #get the document vectors
        document_vectors = [model.infer_vector(
            word_tokenize(doc.lower())) for doc in lstOfText]
        
        #print the document vectors
        for i, doc in enumerate(lstOfText):
            print("Document", i+1, ":", doc)
            print("Vector:", document_vectors[i])
            print()
        
        lstOfVectorEncodedText = document_vectors
        currentVectorEncodedText = document_vectors[-1]
        currentVectorEncodedJSonString = json.dumps(currentVectorEncodedText.tolist())


        conn = get_db_connection()
        conn.execute("INSERT INTO textDatabase (itemNum, textEntered, encodedVector) VALUES (?, ?, ?)", (itemNum, currentText, currentVectorEncodedJSonString))
        for i, encodedVector in enumerate(lstOfVectorEncodedText):
            encodedVectorJsonString = json.dumps(encodedVector.tolist())
            conn.execute("UPDATE textDatabase SET encodedVector = ? WHERE itemNum = ?", (encodedVectorJsonString, i+1))
        conn.commit()
        data = conn.execute('SELECT * FROM textDatabase').fetchall()
        conn.close()

        easierToDisplayData = [(row["itemNum"], row["textEntered"], json.loads(decode_if_bytes(row["encodedVector"]))) for row in data]
        return render_template('results.html', d = easierToDisplayData)"""

@app.route('/results', methods=['POST'])
def routeForResults():
    if request.method == 'POST':
        global promptToGenerateResponsesOn
        promptToGenerateResponsesOn = request.form.get("textEntered")
        global minimumWordCount
        minimumWordCount = int(request.form.get("minWordCount"))
        global maximumWordCount
        maximumWordCount = int(request.form.get("maxWordCount"))


        global currentText
        global lstOfText
        global lstOfVectorEncodedText

        currentText = request.form.get("inputText")

        lstOfText = [
            "Space exploration should continue to be a priority for governments because it drives technological advancements that benefit life on Earth. Many innovations, such as satellite communication, GPS, and advancements in materials science, have emerged from space research. These breakthroughs have widespread applications that impact industries like healthcare, agriculture, and environmental monitoring. Additionally, space exploration fosters international collaboration and the pursuit of knowledge, allowing nations to come together for the greater good and address universal challenges like climate change. Moreover, space exploration is essential for the future of humanity. With Earth's resources being finite and the population growing, exploring other planets and celestial bodies may offer opportunities for resource extraction, colonization, or even survival in the event of a global catastrophe. Investing in space exploration today is an investment in the long-term future of humanity, ensuring we are prepared for challenges that may arise beyond our planet.",
            "While space exploration has provided significant technological advancements, it should not be prioritized over more immediate and pressing issues on Earth. Governments should focus on addressing problems like poverty, climate change, and healthcare rather than spending billions on projects that may not have tangible benefits for people today. The funds spent on space missions could be better allocated to improving quality of life and addressing inequalities worldwide, especially in developing countries where basic needs remain unmet. Furthermore, space exploration, though exciting, is an expensive endeavor that often benefits large corporations or specific scientific fields rather than the general public. The vast amount of resources required for space exploration could be seen as a luxury when many countries face urgent social, economic, and environmental challenges. While the long-term potential of space exploration is undeniable, governments should first ensure that they are addressing the needs of their citizens before investing heavily in space endeavors.",
            "Space exploration should continue to be a priority for governments because it drives innovation and scientific progress. Developing technologies for space missions often leads to breakthroughs that benefit industries on Earth, such as advances in telecommunications, healthcare, and clean energy. For instance, satellite technology, which was initially developed for space exploration, now plays a crucial role in weather forecasting, GPS navigation, and global communication. Moreover, exploring space allows us to address profound scientific questions, such as understanding the origins of life, investigating other planets for potential habitability, and uncovering the mysteries of the universe. Investing in space exploration also fosters international collaboration and inspires future generations. Space missions often bring together scientists and engineers from around the world, strengthening diplomatic relations and uniting humanity in the pursuit of shared goals. Additionally, the achievements of space programs can ignite a passion for science, technology, engineering, and mathematics (STEM) in young minds, encouraging them to pursue careers in these fields. Ultimately, prioritizing space exploration ensures that governments remain at the forefront of innovation, addressing both immediate needs and long-term challenges.",
            "While space exploration is fascinating, it should not take precedence over pressing issues on Earth. Governments face numerous challenges, such as poverty, healthcare crises, climate change, and education inequalities, which demand immediate attention and resources. The high costs of space missions could instead be allocated to addressing these urgent problems. For example, investing in renewable energy initiatives could help combat climate change, while funding public health programs could save countless lives. Prioritizing Earth's challenges over extraterrestrial exploration ensures that governments fulfill their responsibilities to their citizens. Furthermore, space exploration often involves risks and uncertainties, making it a potentially inefficient use of public funds. Many missions have failed or produced limited returns, raising questions about their overall value. Instead of allocating billions to ventures like Mars colonization, governments could focus on improving life on Earth by advancing sustainable agriculture, water management, and global infrastructure. While space exploration has its merits, it should only be pursued once humanity's most critical needs are met, ensuring a more balanced and responsible approach to resource allocation.",
            "Space exploration drives technological innovation and scientific discovery, which have far-reaching benefits for humanity. Many of the technologies we rely on today, such as GPS, satellite communications, and advanced medical imaging, were developed or improved through space research. By investing in space exploration, governments foster advancements that not only enhance our understanding of the universe but also improve life on Earth. Additionally, space exploration inspires future generations to pursue careers in science, technology, engineering, and mathematics (STEM), ensuring a skilled workforce for the future. Furthermore, space exploration addresses existential risks and long-term survival. Earth is vulnerable to threats such as asteroid impacts, climate change, and resource depletion. By exploring space, governments can develop strategies to mitigate these risks, such as asteroid deflection technologies or the potential colonization of other planets. Space exploration also encourages international collaboration, as seen with the International Space Station, fostering peace and cooperation among nations. These efforts underscore the importance of space exploration as a priority for governments, as it safeguards humanity’s future and promotes global unity.",
            "While space exploration has its merits, governments should focus on addressing pressing issues on Earth, such as poverty, healthcare, education, and climate change. These challenges require significant financial resources and immediate attention, and diverting funds to space exploration could delay progress in solving them. For example, the cost of a single space mission could fund numerous social programs or infrastructure projects that directly improve the quality of life for millions of people. Prioritizing Earth-based issues ensures that governments meet their primary responsibility: serving their citizens and improving societal well-being. Moreover, space exploration is often driven by geopolitical competition rather than genuine scientific curiosity, leading to wasteful spending and duplication of efforts. Instead of focusing on space, governments could invest in sustainable technologies and environmental conservation to address climate change, which poses a more immediate threat to humanity. While space exploration may offer long-term benefits, it is a luxury that many nations cannot afford, especially developing countries struggling with basic needs. Redirecting resources toward solving Earth’s problems would create a more equitable and sustainable future for all.",
            "Space exploration should absolutely continue to be a priority for governments because it drives technological innovation and economic growth. The pursuit of space has historically led to breakthroughs that benefit society as a whole—think of satellite technology, which powers global communication, weather forecasting, and GPS systems. Projects like NASA’s Apollo program or SpaceX’s reusable rockets demonstrate how space ambitions push engineering and science forward, often spilling over into other industries. Governments that invest in space aren’t just chasing stars; they’re seeding advancements that can improve life on Earth, from renewable energy solutions to medical technologies inspired by space research. Beyond practical benefits, space exploration taps into a fundamental human need to explore and understand our place in the universe. It’s not just about planting flags on Mars or mining asteroids—it’s about answering big questions: Are we alone? Can humanity survive beyond Earth? These endeavors inspire generations, foster international collaboration, and remind us of what’s possible when we think beyond our planet. In a world facing climate change and resource scarcity, space could also offer long-term solutions, like off-planet habitats or resource extraction. Governments should prioritize it because it’s an investment in both our present and our future.",
            "Space exploration should not remain a government priority when there are pressing issues here on Earth that demand attention and funding. Billions are poured into missions to Mars or distant telescopes while millions lack basic necessities like clean water, healthcare, or education. The argument that space tech trickles down to everyday life is shaky—most innovations could be pursued directly without the astronomical (pun intended) costs of launching rockets. Governments should focus on solving climate change, poverty, and infrastructure decay rather than chasing cosmic dreams that benefit only a small elite of scientists and corporations. Moreover, the risks and uncertainties of space exploration make it a questionable investment. Missions fail, costing taxpayers dearly, and the promise of colonizing other planets remains speculative at best—Earth’s problems won’t vanish by fleeing to Mars. Private companies like SpaceX are already taking up the slack, so why should governments divert resources from immediate human needs to duplicate efforts? Space can wait; it’s been there for billions of years. Right now, the priority should be stabilizing and sustaining the only home we currently have—Earth—before we gamble on the stars.",
            currentText
        ]

        #preprocess the documents, and create TaggedDocuments
        tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                    tags=[str(i)]) for i,
                    doc in enumerate(lstOfText)]
        
        #train the Doc2vec model
        model = Doc2Vec(vector_size=25,
                        min_count=2, epochs=80)
        model.build_vocab(tagged_data)
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        
        #get the document vectors
        document_vectors = [model.infer_vector(
            word_tokenize(doc.lower())) for doc in lstOfText]
        document_vectors_asAnArray = [vector.tolist() for vector in document_vectors]

        """for i, doc in enumerate(document_vectors_asAnArray[0:len(document_vectors_asAnArray)-1]):
            conn = get_db_connection()
            itemNum = i + 1
            conn.execute("INSERT INTO textDatabase (itemNum, textEntered, encodedVector) VALUES (?, ?, ?)", (itemNum, lstOfText[i], doc))               
            conn.commit()"""
        
        
        
        lstOfVectorEncodedText = document_vectors
        currentVectorEncodedText = document_vectors[-1]
        currentVectorEncodedTextAsAnArray = currentVectorEncodedText.tolist()

        squareRootOfCurrentVectorEncodedText = [i*i for i in currentVectorEncodedText]     
        sumOfSquareRootCurrentVectorEncodedText = sum(squareRootOfCurrentVectorEncodedText)
        magnitudeOfInputVector = math.sqrt(sumOfSquareRootCurrentVectorEncodedText)


        cosineSimilarityScores = []

        print(document_vectors_asAnArray[0])
        print(lstOfText)
        


        for paragraph in document_vectors_asAnArray[0:len(document_vectors_asAnArray)-1]:
            dotProduct =  sum([currentVectorEncodedTextAsAnArray[index] * value for index, value in enumerate(paragraph)])
            magnitudeOfParagraph = math.sqrt(sum([i * i for i in paragraph]))
            cosineSimilarityScores.append(dotProduct / (magnitudeOfParagraph * magnitudeOfInputVector))

        

        global highestCosineSimilarityScore
        highestCosineSimilarityScore = max(cosineSimilarityScores)
        indexOfHighestCosineSimilarityScore = cosineSimilarityScores.index(highestCosineSimilarityScore)
        global entryThatMostCloselyMatchesInputText
        entryThatMostCloselyMatchesInputText = lstOfText[indexOfHighestCosineSimilarityScore]
        global threshold
        global aiGenerated
        if(highestCosineSimilarityScore >= threshold):
            aiGenerated = True
        global aiGeneratedAsAString
        aiGeneratedAsAString = "not AI generated"
        if(aiGenerated == True):
            aiGeneratedAsAString = "AI Generated"
        return render_template('secondScreen.html', prompt = promptToGenerateResponsesOn, minWords = minimumWordCount, maxWords = maximumWordCount)

@app.route('/aiGenOrNot')
def routeForAIGenOrNot():
    return render_template("finalScreen.html", closestEntry = entryThatMostCloselyMatchesInputText, AIGEN = aiGeneratedAsAString, thres = threshold, cosSimilarity = highestCosineSimilarityScore, inputText = currentText)

app.debug = True
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
#webpage 
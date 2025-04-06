#import the required libraries
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import numpy as np
import json
import nltk
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

        """lstOfAllDocuments = [
            "Mandatory school uniforms can foster a sense of equality among students by reducing visible socioeconomic differences. When everyone wears the same attire, it minimizes the impact of brand-consciousness or pressure to wear expensive clothing, helping to create a more inclusive environment. This can lead to better interpersonal relationships and reduce bullying based on appearance or financial status, allowing students to focus more on their education. Additionally, school uniforms promote discipline and a sense of identity within the school community. They encourage students to adhere to a code of conduct and represent their institution with pride. Uniforms can also save time for families in the morning, as children don’t have to decide what to wear each day, making the start of the day less stressful. However, critics argue that enforcing uniforms stifles personal expression and individuality. Clothing is often a way for students to showcase their personality, and mandatory uniforms can take away this avenue of self-expression. While they may create uniformity, they don’t address deeper issues related to inequality or peer pressure in other aspects of school life.",
            "Opponents of mandatory school uniforms often highlight the importance of individuality and self-expression. Clothing choices allow students to showcase their unique identities, cultural backgrounds, and creativity. Forcing everyone to wear the same outfit may suppress this critical aspect of personal development, especially during the formative school years. Another argument against uniforms is the potential financial burden they impose on families. Although uniforms are often touted as cost-effective, the specific requirements for certain styles or brands can become expensive. For families with multiple children or limited incomes, this additional expense could be challenging, negating the purported equality uniforms aim to achieve. On the other hand, advocates of uniforms argue that they can minimize distractions and foster discipline. While these benefits are worth considering, critics believe that schools should instead focus on addressing the root causes of behavioral issues or academic performance, rather than implementing a dress code that may only serve as a superficial solution.",
            "The debate over school uniforms often centers on finding the right balance between equality and individuality. Proponents argue that uniforms create a level playing field by reducing socioeconomic disparities and fostering a sense of community. This can enhance focus on academics and extracurricular activities, as students may be less concerned about their appearance or fashion trends. Nevertheless, it’s essential to recognize that mandatory uniforms may not suit every school or student population. In creative or diverse communities, allowing students to wear their own clothes could better reflect their individuality and cultural identity. A compromise, such as a relaxed or partial uniform policy, could accommodate the need for both unity and self-expression. Ultimately, the success of school uniforms depends on thoughtful implementation and feedback from students, parents, and educators. Policies could include options for diverse cultural attire or cost-effective measures for families. By addressing these concerns, schools can aim to achieve a balance that supports both academic excellence and personal growth.",
            "The implementation of school uniforms can promote a sense of discipline and structure within educational environments. When students wear the same attire, it can create a more formal atmosphere that encourages them to take their studies more seriously. Uniforms may also help in reinforcing the idea that school is a place for learning and focus, reducing the likelihood of students getting distracted by casual or inappropriate clothing. This uniformity can foster a sense of belonging among students, making them feel as though they are part of a collective purpose, which may enhance their overall academic experience. However, some argue that uniforms do not address deeper behavioral issues, such as bullying or student engagement. In fact, by focusing on clothing, schools might overlook other aspects of student well-being, like mental health or personal interests. Moreover, students might still find ways to express themselves outside of clothing, such as through hairstyles, accessories, or behaviors. Thus, while uniforms may reduce certain external distractions, they do not necessarily create a more positive learning environment if the underlying issues remain unaddressed. In terms of practicalities, uniforms can offer financial benefits for families. They eliminate the need to buy a wide variety of clothes for school, especially those that follow trendy fashions, which can be expensive. Uniforms ensure that all students are dressed appropriately for school without putting undue pressure on families to keep up with the latest fashion. However, this perceived financial relief might be counterbalanced by the costs of purchasing specific uniforms, especially for families with multiple school-aged children or those who need to buy them annually as their child grows.",
            "School uniforms are often seen as a tool for promoting equality and reducing socioeconomic differences among students. In schools where students come from varying economic backgrounds, uniforms help to level the playing field by ensuring that everyone is dressed the same. This can alleviate the pressure to wear expensive or trendy clothes, which is often a source of bullying or exclusion. By eliminating clothing as a status symbol, uniforms can create a more inclusive environment where students are judged based on their character and academic performance rather than their wardrobe. On the flip side, some argue that uniforms can limit personal freedom and stifle creativity. School is a place where students not only learn academically but also develop their personalities and express their identities. Clothing is a powerful tool for self-expression, and enforcing uniforms may hinder this important aspect of growing up. Critics also point out that uniforms don’t necessarily prevent bullying or cliques; they may just shift the focus to other aspects of difference, such as appearance, mannerisms, or socioeconomic status in other ways that aren't related to clothing. Despite the drawbacks, uniforms can provide practical benefits for parents, particularly when it comes to simplifying morning routines and making school clothing more affordable. Rather than worrying about outfitting their children with an array of school clothes, parents can buy a set of uniforms that can be worn throughout the year. This can ease the burden of constant shopping and reduce the pressure to spend money on expensive brands or trendy pieces. However, the trade-off is that students may not enjoy the flexibility of dressing in a way that reflects their personal style, which may lead to dissatisfaction.",
            "One of the main arguments in favor of school uniforms is that they can create a sense of unity and reduce the visible distinctions between students. In diverse schools, uniforms help diminish the visibility of wealth or fashion status, potentially reducing bullying or peer pressure. When all students wear the same attire, it can foster a feeling of community and equality, making it easier for students to bond over shared experiences rather than superficial differences. This could lead to a more harmonious school environment where students focus on their studies and relationships rather than competing over appearances. Despite these potential benefits, there are concerns that uniforms may not be the magic solution to improving student behavior or academic success. Some critics argue that uniforms fail to address underlying issues, such as bullying or poor mental health, that can affect a student's experience in school. They contend that making uniforms mandatory might distract from more significant reforms needed in education, such as improving teaching quality, mental health resources, or addressing safety issues. Additionally, some students may feel restricted by the uniform, leading to frustration and disengagement. Uniforms could also be seen as a practical way to reduce daily decision fatigue. Students don't have to worry about picking out outfits every morning, which can make the start of the day smoother and more efficient. Parents also save time and money by purchasing a set of uniforms that will last the entire school year. However, this simplicity can come at the cost of students’ sense of autonomy and self-expression. For many, being able to choose what they wear is an important part of developing their independence, and uniforms might undermine that process.",
            "School uniforms should be mandatory because they promote a sense of equality among students. When everyone wears the same attire, it reduces the pressure to keep up with fashion trends or display socioeconomic status through clothing. This can minimize bullying and social exclusion based on appearance, fostering a more inclusive environment where students focus on learning rather than competing over material possessions. One potential benefit is the enhancement of school safety. Uniforms make it easier to identify students belonging to a particular school, which can deter unauthorized individuals from blending in on campus. Additionally, they can reduce gang-related conflicts by eliminating clothing that might signal affiliations, creating a more secure atmosphere for education. However, a drawback is the restriction on self-expression. Students often use clothing as a way to showcase their individuality, and uniforms can stifle this creativity, potentially leading to frustration or disengagement. While the intent is unity, some argue it might come at the cost of personal identity, making it a divisive issue among students and parents alike.",
            "Mandating school uniforms could streamline mornings and reduce decision fatigue for both students and parents. With a set outfit, there’s no need to debate what to wear each day, saving time and reducing stress. This simplicity can help students start their day with a clearer focus on academics rather than wardrobe choices, potentially improving overall productivity. A key benefit is the fostering of discipline and school pride. Uniforms instill a sense of belonging and collective identity, much like a team uniform in sports. This can encourage students to take their roles as learners more seriously and build a stronger connection to their school community, which might translate into better behavior and academic performance. On the flip side, the cost of uniforms can be a burden for some families. While proponents argue they’re cheaper than constantly updating a wardrobe, the initial expense—especially for schools requiring specific, branded items—can strain low-income households. This could ironically deepen inequality rather than alleviate it, countering one of the primary arguments for uniforms.",
            "School uniforms shouldn’t be mandatory because they don’t address the root causes of the issues they aim to solve, like bullying or distraction. While they might level the playing field superficially, determined bullies will find other ways to target peers, such as through accessories or verbal harassment. Forcing uniformity might just mask problems rather than resolve them. A potential benefit, though, is improved focus in the classroom. Without the distraction of flashy or revealing clothing, students and teachers might stay more engaged in lessons. Studies have suggested that uniforms can create a more serious academic tone, which could subtly boost concentration and learning outcomes over time. Yet, a significant drawback is the lack of evidence proving their effectiveness. Research on uniforms often shows mixed results—some schools report better discipline, others see no change. This inconsistency suggests that mandating them might be an unnecessary imposition, especially if the benefits are anecdotal rather than universal, leaving schools to weigh tradition against practicality.",
            "Mandatory school uniforms have been a topic of debate for years, with proponents arguing that they promote equality and reduce distractions. One of the primary benefits is that uniforms can help minimize socioeconomic disparities among students by ensuring everyone dresses similarly, reducing pressure to wear expensive or trendy clothing. Additionally, uniforms can foster a sense of belonging and school pride, creating a more unified and disciplined environment. By removing the focus on fashion, students may also concentrate better on their studies, and uniforms can simplify morning routines for both parents and children. However, there are also significant drawbacks to mandatory uniforms. Critics argue that uniforms stifle individuality and self-expression, which are important aspects of personal development, especially for teenagers. Forcing students to wear the same clothing every day may lead to resentment or a lack of creativity. Furthermore, uniforms can be costly for families, particularly if they are required to purchase multiple sets or specific brands. Some students may also find uniforms uncomfortable or impractical, which could negatively impact their school experience. Ultimately, whether school uniforms should be mandatory depends on the specific context and goals of a school community. While uniforms can promote equality and reduce distractions, they may also limit personal expression and impose financial burdens. Schools should carefully weigh these pros and cons and consider alternative approaches, such as dress codes, which balance uniformity with flexibility. Engaging students, parents, and teachers in the decision-making process can help ensure that any policy adopted meets the needs of the entire community.",
            currentStudentGivenText
        ]"""

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
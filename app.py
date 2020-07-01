import streamlit as st
import pandas as pd
from textblob import TextBlob,Word
import matplotlib.pyplot as plt
import pycountry
from wordcloud import WordCloud , STOPWORDS
import random
import time

st.title('Natural Language Processing App')
st.markdown('Created by - Adarsh Shetty')

st.sidebar.title('Natural Language Processing App')
choice = st.sidebar.selectbox('Select one',options=['Language Detection','Language Translation','Analyse Text'])

languages = {'Kannada':'kn','Arabic':'ar','Armenian':'hy','Bengali':'bn','Burmese':'my','Chineese':'zh',
    'Crotian':'hr','Czech':'cs','Dutch':'nl','English':'en','French':'fr','German':'de','Greek':'el','Gujurati':'gu',
    'Hindi':'hi','Hungarian':'hu','Indonesian':'id','Irish':'ga','Italian':'it','Japaneese':'ja','Korean':'ko','Latin':'la',
    'Malayalam':'ml','Marathi':'mr','Nepali':'ne','Norwegian':'no','Oriya':'or','Persian':'fa','Polish':'pl','Portugeese':'pt','Panjabi':'pa','Romanian':'ro','Russian':'ru',
    'Serbian':'sr','Spanish':'es','Swedish':'sv','Tamil':'ta','Telugu':'te','Ukrainian':'uk','Urdu':'ur','Vietnamese':'vi'}

if choice == 'Language Detection':
    st.subheader('Language Detection App')
    language_detect_text = st.text_area('Enter the text in your langauge')
    if st.button('Check Language'):
        try:
            text_language = TextBlob(language_detect_text)
            lang_code = text_language.detect_language()
            result = pycountry.languages.get(alpha_2=lang_code).name
            st.success(result)
        except:
            st.error('Sorry not able to detect😢')

elif choice == 'Language Translation':
    st.subheader('Language Translation App')
    
    language_translate_text = st.text_area('Enter the text in your langauge')
    language_translate_text = TextBlob(language_translate_text)
    mylang = st.selectbox('Select language to translate',options=list(languages.keys()))
    if st.button('Translate'):
        if language_translate_text == '':
            st.warning('Please enter text to translate')
        else:
            try:
                result = language_translate_text.translate(to=languages[mylang])
                st.success(result)
            except:
                st.warning('Make sure the language to translate and entered language is different')
    
else:
    st.subheader('Analyse Text')
    start = time.time()
    analyse_text = st.text_area('Enter text to Analyse')

    received_text = TextBlob(analyse_text)
    blob_polarity,blob_subjectivity = received_text.sentiment.polarity,received_text.sentiment.subjectivity
    number_of_tokens = len(list(received_text.words))
    nouns = list()
    for word, tag in received_text.tags:
        if tag == 'NN':
            nouns.append(word.lemmatize())
            #len_of_words = len(nouns)
            rand_words = random.sample(nouns,len(nouns))
            final_word = list()
            for item in rand_words:
                word = Word(item)
                final_word.append(word)
                summary = final_word
                end = time.time()
                final_time = end-start
                final_time = round(final_time,5)
    
    if st.button('Submit'):

        st.subheader('Main Points')
        st.write(f'This text has {number_of_tokens} tokens with {len(set(summary))} important points')
        st.markdown('#### Your Text')
        st.info(received_text)
    
        st.write(f'Time Elapsed : {final_time} seconds to analyse')

        st.markdown("### This text is about")
        for i in set(summary):
            st.info(f'[{i}](https://www.dictionary.com/browse/{i}?s=t)')

        st.markdown('#### Sentiment Score')
        data = {"Polarity":[blob_polarity],'Subjectivity':[blob_subjectivity]}
        dataframe = pd.DataFrame(data , columns = ['Polarity','Subjectivity'],index=['Score'])
        dataframe = dataframe.to_html(escape=False)
        st.write(dataframe, unsafe_allow_html=True)
        st.write('  ')

        st.markdown('### Word cloud')
        words = received_text
        processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=600, height=440).generate(processed_words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        st.pyplot()

    if st.button('Clear'):
        pass       

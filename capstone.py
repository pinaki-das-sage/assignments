import speech_recognition as sr
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import nltk
from textblob import TextBlob
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

class Capstone:
    @staticmethod
    def process():
        # use the audio file as the audio source
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("say something")
            audio = r.listen(source)
            print("okay lets try transcribing")

        transcript = ''
        try:
            transcript = r.recognize_google(audio, language='en-IN')
            print("Text = " + transcript)
        except:
            pass

        nltk.download('brown')
        # brown.words()
        # len(brown.words())
        # sent_tokenize(transcript)

        # tokenize the words in the transcribed sentence
        for sent in sent_tokenize(transcript):
            print(word_tokenize(sent))

        nltk.download('stopwords')

        stopwords_en = stopwords.words('english')
        print(stopwords_en)

        # It's a string so we have to them into a set type
        print('From string.punctuation:', type(punctuation), punctuation)

        # Treat the multiple sentences as one document (no need to sent_tokenize)
        # Tokenize and lowercase
        lowered_transcript = list(map(str.lower, word_tokenize(transcript)))
        # print(lowered_transcript)

        stopwords_en = set(stopwords.words('english'))  # Set checking is faster in Python than list.
        stopwords_en_withpunct = stopwords_en.union(set(punctuation))
        # print(stopwords_en_withpunct)

        # List comprehension.
        print([word for word in lowered_transcript if word not in stopwords_en_withpunct])

        nltk.download('averaged_perceptron_tagger')
        # lemmatization

        print('Original transcript: ' + transcript)
        print('\nAfter lemmatization and removing stopwords: ')
        print([word for word in Capstone.lemmatize_sent(transcript)
               if word not in stopwords_en_withpunct and not word.isdigit()])
        blob = TextBlob(transcript)
        # print(blob.tags)
        # print(blob.noun_phrases)
        for sentence in blob.sentences:
            print(sentence.sentiment.polarity)

    # deduce the type of word (verb, noun etc) from its position in the sentence
    def penn2morphy(penntag):
        morphy_tag = {'NN': 'n', 'JJ': 'a',
                      'VB': 'v', 'RB': 'r'}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return 'n'

    def lemmatize_sent(text):
        wnl = WordNetLemmatizer()
        # Text input is string, returns lowercased strings.
        return [wnl.lemmatize(word.lower(), pos=Capstone.penn2morphy(tag))
                    for word, tag in pos_tag(word_tokenize(text))]
import nltk
import random
import string
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import warnings

warnings.filterwarnings('ignore')

# --- download (first time only) ---
#nltk.download('punkt')      # punctuation
#nltk.download('wordnet')    # lexical db
#nltk.download('stopwords')  # stopwords
# --- download (first time only) ---

domain1 = 'domain1.txt'
domain2 = 'domain2.txt'

domain_used = domain1

with open(domain_used) as f:
    doc_full = f.read()

doc_full = doc_full.casefold()

sent_tokens = nltk.sent_tokenize(doc_full)
word_tokens = nltk.word_tokenize(doc_full)
stopwords_pt = nltk.corpus.stopwords.words('portuguese')
lemmatizer = nltk.stem.WordNetLemmatizer()

if domain_used == domain1:
    bot_name = 'RoboCola'
    bot_intent = 'estudar para a PLN'
elif domain_used == domain2:
    bot_name = 'PyBot'
    bot_intent = 'conhecer um pouco mais sobre Python'

# --- normalize text ---
def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

rem_punct = dict((ord(c), None) for c in string.punctuation)

def process(text):
    return lemmatize(nltk.word_tokenize(text.lower().translate(rem_punct)))
# --- normalize text ---

# --- replies and salutation ---
user_salutation = ['oi', 'ola', 'ei', 'oi, tudo bem?', 'tudo bom?', 'tudo bem?']
bot_salutation = ['ola, caro aluno', 'oi', 'seja bem-vindo!']

bot_salutation_q = ['Tudo ótimo! Em que posso ajudá-lo?', 'Por aqui tudo certo! O que posso fazer por você?']

user_farewell = ['só isso', 'tchau', 'flw']
bot_farewell = ['Até logo!', 'Tchau tchau :D', 'Valeu o/']

user_thanks = ['obrigado', 'valeu', 'muito obrigado']

# --- replies and salutation ---
def bot_std_reply(phrase):
    return '%s: %s' % (bot_name, phrase)

def salutation(phrase):
    for word in user_salutation:
        if phrase == word:
            return random.choice(bot_salutation)

def farewell(phrase):
    for word in user_farewell:
        if phrase == word:
            return random.choice(bot_farewell)
# --- replies and salutation ---

# --- term frequency-inverse document frequency ---
def generate_tfidf():
    return TfidfVectorizer(tokenizer=process, stop_words=stopwords_pt)   # tf-idf features

def transform(matrix, params):
    return matrix.fit_transform(params)              # fit and transform

def cos_sim(first_doc, second_doc):
    return cosine_similarity(first_doc, second_doc)  # cosine similarity

def lin_kern(first_doc, second_doc):
    return linear_kernel(first_doc, second_doc)      # linear kernel (faster than cs)
# --- term frequency-inverse document frequency ---

def reply(user_response):
    bot_reply = ''
    sent_tokens.append(user_response)

    term_freq = generate_tfidf()    # initialize
    term_freq_fit = transform(term_freq, sent_tokens)   # convert

    similar_values = linear_kernel(term_freq_fit[-1], term_freq_fit)    # -1 since its appended (input goes to the end)
    similar_vector = similar_values.argsort()[0][-2]    # -2 is the closest item (-1 is the actual item)

    matches = similar_values.flatten()  # condense result array (one dimension)
    matches.sort()

    res_match = matches[-2]

    if res_match == 0:
        bot_reply += 'Ops, não entendi! Pode repetir?'
        return bot_reply

    else:
        bot_reply = bot_reply + sent_tokens[similar_vector]
        return bot_reply

def initialize():
    keep_interaction = True

    print('--- iniciando conversa ---')
    print(bot_std_reply('Eu sou o %s e vou te ajudar a %s :)') % (bot_name, bot_intent))

    while keep_interaction:
        user_response = input('User: ')
        user_response = user_response.lower()

        if user_response in user_thanks:
            print(bot_std_reply('Eu que agradeço! Em que mais posso lhe ser útil?'))

        elif user_response in user_farewell:
            keep_interaction = False
            print(farewell(user_response))

            print('--- finalizando conversa ---')

        else:
            if salutation(user_response) is not None:
                print(bot_std_reply(salutation(user_response)))

            else:
                print(bot_std_reply(''), end='')
                print(reply(user_response))
                sent_tokens.remove(user_response)


start = time.time()
initialize()
end = time.time()

duration = end - start

print('\n--- duração da conversa: %.2f (s) ---' % duration)

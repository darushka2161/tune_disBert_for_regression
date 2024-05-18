from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import string
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import pymorphy2
import numpy as np
import pickle
import pandas as pd
from nltk import ngrams
from collections import Counter
#nltk.download('punkt')
from catboost import CatBoostRegressor

app = Flask(__name__)

with open("UI_for_model\\analysis_base\\one_n_gram.pickle", 'rb') as file_one:
    one_n_gram = pickle.load(file_one)

with open("UI_for_model\\analysis_base\\two_n_gram.pickle", 'rb') as file_one:
    two_n_gram = pickle.load(file_one)

with open("UI_for_model\\analysis_base\\three_n_gram.pickle", 'rb') as file_one:
    three_n_gram = pickle.load(file_one)

with open("UI_for_model\\analysis_base\\conversion.pickle", 'rb') as file_one:
    pseq = pickle.load(file_one)
pseq = pd.DataFrame(pseq).reset_index().iloc[4:-1]


def remove_stopwords(txt):
    rus_stopwords = stopwords.words('russian')
    morph = pymorphy2.MorphAnalyzer(probability_estimator_cls=None)
    
    s = ''
    txt = txt.strip()
    txt = txt.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    txt = re.sub(r'[^\w\s]|\n', ' ', txt)
    txt = txt.lower()
    txt = re.sub('[^а-яА-ЯёЁ*\W]',' ',txt)
    for word in txt.split():
        word = morph.parse(word)[0].normal_form
        if word not in rus_stopwords:
            if word not in ['также', 'весь', 'это', 'который', 'иза', 'еще', 'ещё', 'ее', 'её', 'свой']:
                s = s+ word + ' '
    s = s[:-1]
    return s



def count_emb(text):
    tokenizer = AutoTokenizer.from_pretrained('DARUSHKA2161/distilbert-base-multilingual-cased-vk-posts')
    model = AutoModel.from_pretrained('DARUSHKA2161/distilbert-base-multilingual-cased-vk-posts')
    encoding = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask = True,
            return_tensors ='pt')
    outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
    return outputs.last_hidden_state[0,0,:].tolist()


def closest_vector(target_vector):
    with open("C:\\Users\\Dasha\\Desktop\\ВКР\\post_with_emb.pickle", 'rb') as file_one:
        posts = pickle.load(file_one)
    posts = posts[posts['конверсия'] > 0.003523]
    dataset = list(posts['emb'])
    cosine_distances = np.dot(dataset, target_vector) / (np.linalg.norm(dataset, axis=1) * np.linalg.norm(target_vector))
    posts['cos_distance'] = cosine_distances
    posts = posts[posts['cos_distance']<0.999999].reset_index(drop = True)
    closest_vector_index = np.argmax(posts['cos_distance'])
    final_string = posts.text.iloc[closest_vector_index]
    final_string = str(final_string)
    return final_string.replace('\\n', ' ').replace('\n', ' ')


def model_prediction(text, text_base):
    tokenizer = AutoTokenizer.from_pretrained('DARUSHKA2161/distilbert-base-multilingual-cased-vk-posts')
    model = AutoModelForSequenceClassification.from_pretrained('DARUSHKA2161/distilbert-base-multilingual-cased-vk-posts', num_labels = 1)
    encoding = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask = True,
            return_tensors ='pt')
    outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask']).logits.tolist()[0][0]

    get_features = pd.concat([one_n_gram.sort_values('freq_dif').head(100),
            one_n_gram.sort_values('freq_dif').tail(100),
            two_n_gram.sort_values('freq_dif').head(100),
            two_n_gram.sort_values('freq_dif').tail(100),
            three_n_gram.sort_values('freq_dif').head(100),
            three_n_gram.sort_values('freq_dif').tail(100)])
    
    df_features = pd.DataFrame(columns = [str(i) for i in get_features['n_gram']])
    df_features['label'] = [outputs]
    df_features['len_text'] = [len(text_base.split())]
    
    text_frame = pd.concat([get_dataframe_freq(1, text), get_dataframe_freq(2, text), get_dataframe_freq(3, text)])
    columns_put = [str(i) for i in pd.merge(get_features, text_frame, on = 'n_gram')['n_gram']]
    for col in columns_put:
        df_features.loc[0, col] = 1

    df_features = df_features.fillna(0)

    with open("UI_for_model\\analysis_base\\cat_model.pickle", 'rb') as file_one:
        model = pickle.load(file_one)
    
    predictions = model.predict(df_features)
    predictions = predictions[0]

    persent = pseq[pseq['конверсия'] <= predictions].iloc[-1]['index']


    return str(round(predictions, 6)) + " (конверсия от твоего поста может быть выше, чем у " + persent + " постов)" 



def get_dataframe_freq(n, text):
    def count_ngrams(tokens, n):
        ngrams_list = ngrams(tokens, n)
        return Counter(ngrams_list)

    tokens = nltk.word_tokenize(text.lower())
    ngram_freq = count_ngrams(tokens, n)
    df = pd.DataFrame(data = list(ngram_freq.items()), columns = ['n_gram', 'freq'])
    df = df.sort_values('freq')
    return df



def get_colored_text(text_norm, text):
    morph = pymorphy2.MorphAnalyzer(probability_estimator_cls=None)

    get_down= pd.concat([one_n_gram.sort_values('freq_dif').tail(100),
            two_n_gram.sort_values('freq_dif').tail(100),
            three_n_gram.sort_values('freq_dif').tail(100)])

    get_up = pd.concat([one_n_gram.sort_values('freq_dif').head(100),
            two_n_gram.sort_values('freq_dif').head(100),
            three_n_gram.sort_values('freq_dif').head(100)])

    text_frame = pd.concat([get_dataframe_freq(1,text_norm), get_dataframe_freq(2, text_norm), get_dataframe_freq(3, text_norm)])

    n_grams_up = pd.merge(get_up, text_frame, on = 'n_gram')
    set_up_word = set()
    for item in n_grams_up.n_gram:
        set_up_word = set_up_word|set(item)

    n_grams_down = pd.merge(get_down, text_frame, on = 'n_gram')
    set_down_word = set()
    for item in n_grams_down.n_gram:
        set_down_word = set_down_word|set(item)

    finish_str = ''
    for post_text in text.split():
        post_text_cut=re.sub(r'[^\w\s]|\n', '',post_text)
        if morph.parse(post_text_cut)[0].normal_form in set_up_word:
            finish_str = finish_str+'<span class="green-box">' + post_text + '</span>' +' '
        elif morph.parse(post_text_cut)[0].normal_form in set_down_word:
            finish_str = finish_str+'<span class="pink-box">' + post_text + '</span>' +' '
        else:
            finish_str = finish_str+'<span class="info-value">' + post_text + '</span>' +' '

    return finish_str[:-1]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form['user_text']

        clean_text = remove_stopwords(user_text)
        final_number = model_prediction(clean_text, user_text)
        emb = count_emb(clean_text)
        similar_text = closest_vector(emb).replace('\\n', ' ').replace('\n', ' ').replace('\\', ' ')
        text_res = get_colored_text(clean_text, user_text).replace('\\n', ' ').replace('\n', ' ').replace('\\', ' ')
        return render_template('result.html',start_text = text_res,  final_number=final_number, similar_text=similar_text)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
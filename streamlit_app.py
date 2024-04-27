import streamlit as st
import torch
import os
import torch.nn as nn
import numpy as np
import torchtext
import pyperclip
from numpy import dot
from numpy import average
from numpy.linalg import norm
from autocorrect import Speller
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import builtins

nltk.download('stopwords')


class context_getter:
    def __init__(self):
        try:
            self._glove = st.session_state.glove
        except Exception:
            st.session_state.glove = self._load_glove_vectors()
            self._glove = st.session_state.glove

        self._sense_vectors_collection = {}

    def _load_glove_vectors(self):
        with st.spinner('Loading glove vectors...'):
            torchtext.vocab.GloVe(name='twitter.27B', dim=50)
            glove_file = os.path.join(os.getcwd(), '.vector_cache\\glove.twitter.27B.50d.txt')
            f = open(glove_file, 'r', encoding="utf-8")
            vectors = {}
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                vectors[word] = embedding
            f.close()
            return vectors

    def _get_valid_pos_tag(self, tag):
        if tag.startswith('J') or tag.startswith('V') or tag.startswith('N') or tag.startswith('R'):
            return True
        return False

    def _get_word_sense_vectors(self, candidate):
        cosine_sim_threshold = 0.05

        vectors = {}
        try:
            candidate_vec = self._glove[candidate]
        except Exception:
            return None
        for sense in wn.lemmas(candidate):
            gloss = [sense.synset().definition()]
            gloss.extend(sense.synset().examples())
            word_vectors = []
            for sentence in gloss:
                tokens = nltk.word_tokenize(sentence)
                pos_tags = nltk.pos_tag(tokens)
                for gloss_pos, tag in pos_tags:
                    if self._get_valid_pos_tag(tag):
                        try:
                            gloss_word_vec = self._glove[gloss_pos]
                        except Exception:
                            # print(gloss_pos, "not found in glove")
                            continue
                        cos_sim = dot(gloss_word_vec, candidate_vec) / (norm(gloss_word_vec) * norm(candidate_vec))
                        if cos_sim > cosine_sim_threshold:
                            word_vectors.append(gloss_word_vec)
            if len(word_vectors) == 0:
                continue
            sense_vector = average(word_vectors, 0)
            vectors[sense] = sense_vector
        return vectors

    def _disambiguate_word_sense(self, word, context_vector):
        vectors = self._sense_vectors_collection[word]
        if len(vectors) == 0:
            return [None, 0.0]
        cos_sims = {}
        for sense, sense_vector in vectors.items():
            cos_sim = dot(context_vector, sense_vector) / (norm(context_vector) * norm(sense_vector))
            cos_sims[sense] = cos_sim
        sorted_list = sorted(cos_sims.items(), key=lambda x: x[1])
        if len(sorted_list) == 0:
            return [None, 0.0]
        most_similar_pair = sorted_list.pop()
        disambiguated_sense = most_similar_pair[0]
        cos_sim_second_most_similar_sense = 0
        if len(sorted_list) > 0:
            cos_sim_second_most_similar_sense = sorted_list.pop()[1]
        score_margin = most_similar_pair[1] - cos_sim_second_most_similar_sense
        # we return the disambiguated sense AND the cosine score margin between the two most similar senses.
        return [disambiguated_sense, score_margin]

    def run_algorithm(self, tokens_input):
        score_margin_threshold = 0.1

        sorted_sense_vectors_collection = {}
        pos_tags_input = nltk.pos_tag(tokens_input)

        pos = []
        pos_vectors = {}
        for word, pos_tag in pos_tags_input:
            if self._get_valid_pos_tag(pos_tag):
                try:
                    pos_vectors[word] = self._glove[word]
                    pos.append(word)
                except Exception:
                    pass

        # Sense vectors init
        for p in pos:
            sense_vectors = self._get_word_sense_vectors(p)
            if sense_vectors is None:
                continue
            self._sense_vectors_collection[p] = sense_vectors
            sorted_sense_vectors_collection[p] = len(sense_vectors)

        # S2C sorting for content word
        sorted_sense_vectors_collection = sorted(sorted_sense_vectors_collection.items(), key=lambda x: x[1])

        # Context vector initialization
        context_vec = average(list(pos_vectors.values()), 0)

        for w, _ in sorted_sense_vectors_collection:
            disambiguation_results = self._disambiguate_word_sense(w, context_vec)
            disambiguated_sense = disambiguation_results[0]
            if disambiguated_sense is None:
                continue
            score_margin = disambiguation_results[1]
            if score_margin > score_margin_threshold:
                pos_vectors[w] = self._sense_vectors_collection[w][disambiguated_sense]
                context_vec = average(list(pos_vectors.values()), 0)
        self._sense_vectors_collection.clear()
        return context_vec


class ToneTagsLSTM_wsd_2(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_dim, context_dim, output_size, num_layers, dropout):
        super(ToneTagsLSTM_wsd_2, self).__init__()

        # output_size = 19

        self.embedding = embedding

        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True,
                            dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim * 4096 * 2 + context_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)
        # self.out = nn.Softmax(output_size, dim=1)

    def forward(self, tokens, contexts):
        embedded = self.embedding(tokens)
        output, (hidden, cell) = self.lstm(embedded)

        lstm_out = torch.cat((output.reshape(1, -1).to(torch.float32), contexts.to(torch.float32)), dim=1)

        fc1_out = self.fc1(lstm_out)

        fc2_out = self.fc2(fc1_out)
        out = self.fc3(fc2_out)
        # out = self.out(fc3_out)

        return out


class main:

    def _get_text_tokens_ids(self, text_tokens):
        with st.spinner('Loading glove vocab...'):
            try:
                vocab = st.session_state.glove_twitter_27B.stoi
            except Exception:
                st.session_state.glove_twitter_27B = torchtext.vocab.GloVe(name='twitter.27B', dim=50)
                vocab = st.session_state.glove_twitter_27B.stoi
                vocab["<unk>"] = len(vocab)
                vocab["<pad>"] = len(vocab)

        max_length = 4096

        if len(text_tokens) > max_length:
            st.error('Error: Text is very long: Cut it')
            return None
        else:
            # transform tokens to ids
            text_ids = []
            for token in text_tokens:
                try:
                    text_ids.append(vocab[token])
                except Exception:
                    text_ids.append(vocab["<unk>"])

            # add padding ids to correct model work
            while len(text_ids) < max_length:
                text_ids.append(vocab["<pad>"])

            return text_ids

    def _get_context_preprocess_text_to_ids(self, text):
        with st.spinner('Correcting text...'):
            # correct text
            spell = Speller()
            corrected_text = spell(text).lower()

        with st.spinner('Removing stopwords...'):
            # remove stopwords with nltk stopwords
            word_tokens = word_tokenize(corrected_text)
            stop_words = set(stopwords.words('english'))
            text_tokens = [w for w in word_tokens if not w in stop_words]

            # st.write('Corrected text:')
            # st.write('{}'.format(' '.join(text_tokens)))

        with st.spinner('Getting context and tokens of text...'):
            # get context vector from text_tokens by glove.twitter.27B.50d
            context_getter_instance = context_getter()
            context = context_getter_instance.run_algorithm(text_tokens)

        with st.spinner('Converting tokens to ids...'):
            # preprocess text_tokens to text_ids with vocab by glove.twitter.27B.50d
            text_ids = self._get_text_tokens_ids(text_tokens)
            if text_ids is None:
                return None

        return torch.tensor([text_ids]), torch.tensor([context])

    def _load_torch_model(self, device):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'results\\models\\lstm_model.pt')

        model = torch.load(model_path, map_location=torch.device(device))
        return model

    def _predict_tone_tags(self, text_ids, context):
        with st.spinner('Downloading model...'):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                st.session_state.model.to(device)
            except Exception:
                st.session_state.model = self._load_torch_model(device)

        with st.spinner('Predicting Tone Tags...'):
            with torch.no_grad():
                st.session_state.model.to(device)
                st.session_state.model.eval()

                text_ids = text_ids.to(device)
                context = context.to(device)

                predictions = st.session_state.model(text_ids, context)

            probabilities = torch.softmax(predictions, dim=1).tolist()[0]
            labels = ['genuine question', 'half joking', 'genuine', 'not a vent', 'reference', 'serious', 'platonic',
                      'inside joke', 'sarcastic', 'joking', 'romantic', 'passive aggressive', 'copypasta', 'ironic',
                      'clickbait', 'lyrics', 'nothing personal', 'not mad', 'rhetorical']
            tone_tags_probs = dict(zip(labels, probabilities))
            sorted_tone_tags_probs = dict(sorted(tone_tags_probs.items(), key=lambda item: item[1], reverse=True)[:3])
            return sorted_tone_tags_probs

    def _check_text_tone_tags(self, text):
        if text == '':
            st.error('Error: Empty text.')
        else:
            # preprocess text and get context of it
            with st.spinner('Preprocessing text...'):
                text_ids_context = self._get_context_preprocess_text_to_ids(text)
                # st.write(text_ids_context)

                if text_ids_context is None:
                    return None

                try:
                    if pd.isna(text_ids_context[1][0].tolist()).any():
                        return None
                except Exception:
                    if pd.isna(text_ids_context[1][0].tolist()):
                        return None

                text_ids, context = text_ids_context

            # predict tone tags probabilities
            with st.spinner('Predicting...'):
                tone_tags_probs = self._predict_tone_tags(text_ids, context)

            return tone_tags_probs

    def main(self):

        try:
            st.session_state.labels_to_tone_tag.keys()
        except Exception:
            st.session_state.labels_to_tone_tag = {'genuine question': '/genq', 'half joking': '/hf', 'genuine': '/g, /gen', 'not a vent': '/nav', 'reference': '/ref', 'serious': '/srs', 'platonic': '/p',
                      'inside joke': '/ij', 'sarcastic': '/s', 'joking': '/j', 'romantic': '/r', 'passive aggressive': '/pa', 'copypasta': '/c', 'ironic': '/iron',
                      'clickbait': '/cb', 'lyrics': '/l, /ly', 'nothing personal': '/np', 'not mad': '/nm', 'rhetorical': '/rh, /rt'}

        st.header(f"Check tone tag of your text")
        users_text = st.text_area("Write your text here:", height=300)
        if users_text:
            st.session_state.users_text = users_text
            st.session_state.tone_tags_probs = self._check_text_tone_tags(users_text)

        # if st.button("Run it"):
        #     st.session_state.users_text = users_text
        #     st.session_state.tone_tags_probs = self._check_text_tone_tags(users_text)

        try:
            if st.session_state.tone_tags_probs is not None:
                st.write("Choose your tone tag and click button to copy it to clipboard:")
                for tone_tag in st.session_state.tone_tags_probs.items():
                    st.button(f" {round(tone_tag[1] * 100)}% - {tone_tag[0]}")
                    pyperclip.copy(f"{st.session_state.users_text} {st.session_state.labels_to_tone_tag[tone_tag[0]]}")
            else:
                st.error('Error: Write correct english text.')
        except Exception:
            pass

if __name__ == "__main__":
    main = main()
    main.main()

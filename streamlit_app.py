import streamlit as st
import requests
import torch
import os
import torch.nn as nn

class ToneTagsLSTM_wsd_2(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_dim, context_dim, output_size, num_layers, dropout):
        super(ToneTagsLSTM_wsd_2, self).__init__()

        # output_size = 19

        self.embedding = embedding

        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim * 4096 * 2 + context_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)
        # self.out = nn.Softmax(output_size, dim=1)


    def forward(self, tokens, contexts):

        embedded = self.embedding(tokens)
        output, (hidden, cell) = self.lstm(embedded)

        lstm_out = torch.cat((output.reshape(32, -1), contexts), dim=1)

        fc1_out = self.fc1(lstm_out)


        fc2_out = self.fc2(fc1_out)
        out = self.fc3(fc2_out)
        # out = self.out(fc3_out)

        return out


def check_text_tone_tag(text):
    return "Tested tone tag"


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'results\models\lstm_model.pt')

    with st.spinner('Downloading model...'):
        model = torch.load(model_path)


    st.header(f"Check tone tag of your message/text")

    users_text = st.text_area("Write your text here:", height=500, max_chars=500000)

    if st.button("Run it"):
        tone_tag = check_text_tone_tag(users_text)
        st.write(f"{tone_tag}")


if __name__ == "__main__":
    main()

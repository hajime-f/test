"""Livedoorニュースを分類するプログラム"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, T5Tokenizer


class NewsClassifier(nn.Module):

    def __init__(self):
        super(NewsClassifier, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
        self.rinna = AutoModelForCausalLM.from_pretrained(
            "rinna/japanese-gpt-1b")
        self.classifier = nn.Linear(self.rinna.lm_head.out_features, 9)

    def forward(self, input_ids):
        output = self.rinna(input_ids=input_ids)
        return self.classifier(output[0][:, -1, :])


if __name__ == '__main__':

    # データの読み込み
    data = pd.read_csv('livedoor_news.csv')

    # データの分割
    train, test = train_test_split(data, test_size=0.2, random_state=0)

    # モデルの定義
    model = NewsClassifier()

    # 最適化手法の定義
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 損失関数の定義
    loss_function = nn.CrossEntropyLoss()

    # 学習
    model.train()
    for epoch in range(5):
        for _, row in train.iterrows():

            xtrain = row['words'][:1000]
            ytrain = torch.tensor([row['category']])
            print(xtrain)
            ids = model.tokenizer.encode(
                xtrain, add_special_tokens=False, return_tensors="pt")
            outputs = model(ids)
            loss = loss_function(outputs, ytrain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        for _, row in test.iterrows():

            xtest = row['words'][:1000]
            ytest = torch.tensor([row['category']])

            ids = model.tokenizer.encode(
                xtest, add_special_tokens=False, return_tensors="pt")

            outputs = model(ids)
            ans = torch.argmax(outputs, 1)

            breakpoint()

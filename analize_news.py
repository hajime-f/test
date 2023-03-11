"""Livedoorニュースを分類するプログラム"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, T5Tokenizer

if __name__ == '__main__':

    # データの読み込み
    data = pd.read_csv('livedoor_news.csv')

    # データの分割
    data_train = data.iloc[:5000, :]
    data_test = data.iloc[5000:, :]

    # モデルの読み込み
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")
    if torch.cuda.is_available():
        model = model.to("cuda")

    # モデルの構造を変更（最終レイヤー関数の付け替え）
    in_features = model.lm_head.in_features
    model.lm_head = nn.Linear(in_features, 9)

    # 損失関数を定義
    criterion = nn.CrossEntropyLoss()

    # 最適化手法を定義
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 学習用データを準備
    for i, row in data_train.iterrows():
        token_ids = tokenizer.encode(
            row['words'], add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_length=100,
                min_length=100,
                do_sample=True,
                top_k=500,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        breakpoint()

    # 学習
    model.train()
    for epoch in range(10):

        breakpoint()

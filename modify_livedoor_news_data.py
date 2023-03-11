import pandas as pd

if __name__ == '__main__':

    # データの読み込み
    data = pd.read_csv('livedoornews.csv')

    # 空のデータフレームを用意する
    df = pd.DataFrame(columns=['words', 'category'])

    for i in range(len(data)):

        title = data.title[i].replace('\n', '').replace('\u3000', '')
        body = data.body[i].replace('\n', '').replace('\u3000', '')

        if data.media[i] == 'movie-enter':
            category = 0
        elif data.media[i] == 'it-life-hack':
            category = 1
        elif data.media[i] == 'kaden-channel':
            category = 2
        elif data.media[i] == 'topic-news':
            category = 3
        elif data.media[i] == 'livedoor-homme':
            category = 4
        elif data.media[i] == 'peachy':
            category = 5
        elif data.media[i] == 'sports-watch':
            category = 6
        elif data.media[i] == 'dokujo-tsushin':
            category = 7
        elif data.media[i] == 'smax':
            category = 8
        else:
            category = -1

        df_append = pd.DataFrame(data={'words': [title + body],
                                       'category': [category]})
        df = pd.concat([df, df_append], ignore_index=True)

    # データの保存
    df.to_csv('livedoor_news.csv', index=False)

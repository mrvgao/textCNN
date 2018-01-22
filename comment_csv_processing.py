import pandas as pd
from tqdm import tqdm
import jieba


def cut(string): return list(jieba.cut(string))


movie_comment = pd.read_csv('data/movie_comments.csv', encoding='utf-8')

pos_comment = open('data/douban_comments/db_positive_words.txt', 'w', encoding='utf-8')
neg_comment = open('data/douban_comments/db_negative_words.txt', 'w', encoding='utf-8')

test_mode, steps = False, 0

max_size = 50
min_size = 10

for data in tqdm(movie_comment.iterrows(), total=100 if test_mode else len(movie_comment)):
    comment = str(data[1]['comment'])
    star = data[1]['star']
    try:
        star = int(star)
    except ValueError:
        continue

    if len(comment) < min_size or len(comment) > max_size: continue

    if star <=2:
        # negative
        repeat_num = 2 if steps % 2 == 0 else 3
        for i in range(repeat_num):
            neg_comment.write(' '.join(cut(comment)) + '\n')
    elif star >= 4:
        # positive
        pos_comment.write(' '.join(comment) + '\n')

    if test_mode and steps > 100: break
    steps += 1

pos_comment.close()
neg_comment.close()


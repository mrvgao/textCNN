from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

vectorized = TfidfVectorizer()


def get_tfidf_vector(corpus):
    return vectorized.fit_transform(corpus)


if __name__ == '__main__':
    lines = []
    for ii, line in enumerate(open('data/douban_comments/db_negative_words.txt', encoding='utf-8')):
        if ii > 10000: break
        lines.append(line)

    X = get_tfidf_vector(lines)[:10]
    print(X[:10])

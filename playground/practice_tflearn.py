import tflearn

lines = [l for l in open('../data/douban_comments/db_negative.txt', encoding='utf-8')]
max_size = max([len(line.split()) for line in lines])
vocabulay_process = tflearn.data_utils.VocabularyProcessor(max_document_length=max_size)
line = list(vocabulay_process.fit_transform(lines))

print(line)
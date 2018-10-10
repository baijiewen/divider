from ..models import TrainingSet
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from .. import db


def preprocess(text):
    text_with_spaces = ""
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + " "
    return text_with_spaces


def loadtrainset(db_type, classtag):
    cut_text = TrainingSet.query.filter_by(db_type=db_type, classtag=classtag, is_cut=True).all()
    processed_textset = []
    allclasstags = []
    for text in cut_text:
        processed_textset.append(text)
        allclasstags.append(classtag)
    return processed_textset, allclasstags


def get_need_trainning_set():
    new_text = TrainingSet.query.filter_by(is_cut=False).first()
    if new_text is None:
        return False,False
    cut_text = preprocess(new_text.text)
    new_text.cut_text = cut_text
    new_text.is_cut = True
    return cut_text,new_text.id


def update_text(text_id, predict_result):
    text = TrainingSet.query.get_or_404(text_id)
    text.db_type = predict_result[0]
    text.classtag = predict_result[1]
    db.session.add(text)
    db.session.commit()


def training():
    all_kinds = TrainingSet.query.filter_by(is_cut=False).group_by("kind").all()
    for kind in all_kinds:
        #classtags_list,integrated_train_data #todo
        continue
    integrated_train_data, classtags_list = [], []
    count_vector = CountVectorizer()
    vector_matrix = count_vector.fit_transform(integrated_train_data)
    train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)
    clf = MultinomialNB().fit(train_tfidf, classtags_list)
    testset = []
    cut_text, text_id = get_need_trainning_set()
    if cut_text:
        testset.append(cut_text)
        new_count_vector = count_vector.transform(testset)
        new_tfidf = TfidfTransformer(use_idf=False).fit_transform(new_count_vector)
        predict_result = clf.predict(new_tfidf)
        print(predict_result)
        update_text(text_id, predict_result)
    else:
        print('not need traning')


if __name__ == '__main__':
    training()
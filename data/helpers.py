from dhivehi_nlp.stemmer import stem
from dhivehi_nlp.tokenizer import word_tokenize
from dhivehi_nlp.stopwords import get_stopwords


stop_words = get_stopwords()
def cleaner(text):
    text = "".join([i for i in text if not i.isdigit()])
    stemed_text = stem(text)
    word_tokens = word_tokenize(" ".join(stemed_text), removeNonDhivehiNumeric=True, removePunctuation=True)

    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)
    text = (" ".join(filtered_sentence))
    return text
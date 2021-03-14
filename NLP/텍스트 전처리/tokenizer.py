from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag

from konlpy.tag import Okt  

import nltk
import kss



class WordTokenizer():
    def nltk_word_tokenize(self, text):
        return word_tokenize(text)

    def nltk_WordPunctTokenizer(self, text):
        return WordPunctTokenizer().tokenize(text)
    
    def nltk_TreebankWordTokenizer(self, text):
        return TreebankWordTokenizer().tokenize(text)

    def konlpy_okt(self, text):
        return Okt().morphs(text)
    
    def nltk_pos_tag(self, token):
        return pos_tag(token)
    

class SentenceTokenizer():
    def nltk_sent_tokenize(self, text):
        return sent_tokenize(text)
    
    def kss_sent_tokenize(self, text):
        return kss.split_sentences(text)

if __name__ ==  "__main__":
    word_token = WordTokenizer()
    sentence_token = SentenceTokenizer()
    text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
    sentence = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
    k_sentence = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'

    # 서로 다른 토크나이져 상황에 따라 성능이 좋은 토크나이져 사용
    print(f"word_tokenize : {word_token.nltk_word_tokenize(text)}")
    print(f"WordPunctTokenizer : {word_token.nltk_WordPunctTokenizer(text)}")
    print(f"TreebankWordTokenizer : {word_token.nltk_TreebankWordTokenizer(text)}")
    print(f"konlpy_okt : {word_token.konlpy_okt(k_sentence)}")
    
    token = word_token.nltk_word_tokenize(text)
    #문장 토크나이져 NLTK, OpenNLP, 스탠포드 CoreNLP, splitta, LingPipe 등등이 있다.
    print(f"nltk_sent_tokenize : {sentence_token.nltk_sent_tokenize(sentence)}")
    print(f"kss_sent_tokenize : {sentence_token.kss_sent_tokenize(k_sentence)}")

    # 품사 태깅 => 같은 단어라도 품사에 따라 다른 뜻을 가질수 있기 때문
    print(f"nltk_pos_tag : {word_token.nltk_pos_tag(token)}")
    
    
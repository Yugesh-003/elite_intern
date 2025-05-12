from nltk import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('wordnet')  
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger') 

from nltk.stem.wordnet import WordNetLemmatizer 

sentence = "I ate apple"

wordtoken = word_tokenize(sentence)
print(wordtoken)

wordlist = []
lemma = WordNetLemmatizer()

for word in wordtoken:
    l = lemma.lemmatize(word)
    wordlist.append(l)

print(wordlist)

sent = ' '.join(wordlist)
print(sent)

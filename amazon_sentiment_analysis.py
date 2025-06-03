#!/usr/bin/env python
# coding: utf-8

# In[2]:


# If not installed already, run once:
# !pip install pandas nltk scikit-learn matplotlib wordcloud

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[3]:


get_ipython().system('pip install wordcloud')


# In[4]:


import sys
get_ipython().system('{sys.executable} -m pip install wordcloud')


# In[5]:


python3 -m pip install wordcloud


# In[6]:


get_ipython().system('pip install wordcloud')


# In[7]:


python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install wordcloud


# In[1]:


# 1. Imports and Downloads
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# In[2]:


# 1. Imports and Downloads
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# In[3]:


# 2. Load Data
data = pd.read_csv('/Users/medepatiramtejareddy/Downloads/Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv')
data.dropna(inplace=True)  # remove nulls
data.head()


# In[4]:


# 3. Prepare Sentiment labels
data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1
data['Sentiment'].value_counts()


# In[5]:


# 4. Clean reviews: remove stopwords
stop_words = set(stopwords.words('english'))

def clean_review(text):
    return " ".join(word for word in text.split() if word.lower() not in stop_words)

data['Review'] = data['Review'].apply(clean_review)
data.head()


# In[6]:


# 5. Wordclouds for negative reviews (Sentiment=0)
neg_text = " ".join(data[data['Sentiment'] == 0]['Review'])
plt.figure(figsize=(15,10))
wordcloud = WordCloud(width=1600, height=800, max_font_size=110, random_state=21).generate(neg_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[7]:


# 7. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=2500)
X = tfidf.fit_transform(data['Review']).toarray()
y = data['Sentiment']


# In[8]:


# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[9]:


# 9. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[10]:


# 10. Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')


# In[11]:


# 11. Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.show()


# In[ ]:





# %%
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import joblib
import string 

# %%
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# %%
fake.head()

# %%
true.head()

# %%
fake['class'] = 0
true['class'] = 1

# %%
data = pd.concat([fake, true], axis = 0)

# %%
data.sample(10)

# %%
data = data.drop(["title", "subject", "date"], axis = 1)

# %%
data.reset_index(inplace=True)

# %%
data.drop(["index"], axis = 1, inplace=True)

# %%
data.sample(5)

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r"\\w", "", text)
    text = re.sub(r"https?:://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

# %%
data["text"] = data["text"].apply(clean_text)

# %%
x = data["text"]
y = data["class"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)

# %%
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(xtrain)
xv_test = vectorizer.transform(xtest)

# %%
lr = LogisticRegression()
lr.fit(xv_train, ytrain)

# %%
prediction = lr.predict(xv_test)
lr.score(xv_test,ytest)

# %%
print(classification_report(ytest, prediction))

# %%
joblib.dump(vectorizer, "vectorizer.jb")
joblib.dump(lr, "lr_model.jb")

# %%




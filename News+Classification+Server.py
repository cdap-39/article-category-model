import json
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification

twenty_train = load_files('./news/20news-bydate-train',description=None, categories=None, load_content=True, shuffle=True, encoding='utf8',decode_error='ignore')
# twenty_train = bunch(subset='train', shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(twenty_train.data, twenty_train.target, test_size=.1)


# a = 'President Trump moved on Friday to leave an even deeper mark on Republican primary season, boosting a personal ally who is running for governor of Florida and extending political clemency to a former critic, Representative Martha Roby of Alabama, who is in a difficult race for re-election.Ms. Roby has faced criticism from the right since withdrawing her endorsement of Mr. Trump in the closing weeks of the 2016 presidential election, after the release of the “Access Hollywood” recording that showed Mr. Trump boasting about groping women. Alabama Republicans declined to re-nominate her in a primary election earlier this month, forcing her instead into a July 17 runoff vote.'
# b = 'Sri Lanka Captain Dinesh Chandimal, Coach Chandika Hathurusinghe and Manager Asanka Gurusinha have admitted to breaching Article 2.3.1, a Level 3 offence, which relates to “conduct that is contrary to the spirit of the game.Following their admission, the ICC, in accordance with Article 5.2 of the ICC Code of Conduct, has appointed The Hon Michael Beloff QC as the Judicial Commissioner to hear the case to determine the appropriate sanction.'

for i in range(len(twenty_train.target_names)):
          print(twenty_train.target_names[i])  # prints all the categories

# print(X_test[:1])  # prints first line of the first data file


# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline

clf = MultinomialNB().fit(X_train_tfidf, y_train)
# # Building a pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print("Naive Bayes Accuracy :" + str(np.mean(predicted == y_test)))

# SGDClassifier (SVM)
from sklearn.linear_model import SGDClassifier
# twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
# np.mean(predicted == y_test)

# predicted = text_clf.predict(X_test)
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='modified_huber',penalty = 'elasticnet', alpha=1e-3, n_iter=5, random_state=42))])
# random forest
# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
# Rclf = RandomForestClassifier(max_depth=2, random_state=0)
# Rclf.fit(X_train, y_train)
# predicted_RF = Rclf.predict(X_test)


# print("Random forest Accuracy :" + str(np.mean(predicted_RF == y_test)))
# modified_huber
text_clf_svm = text_clf_svm.fit(X_train, y_train)
predicted_svm_a = text_clf_svm.predict(X_test)
print("ML work done.")
print("SVM Accuracy :" + str(np.mean(predicted_svm_a == y_test)))

def getCategories(newPredict):
    # print(newPredict)  # prints all the data
    predicted_svm = text_clf_svm.predict(newPredict)
    predicted_nvb = text_clf.predict(newPredict)
    print("this test values")
    print(newPredict)
    print(predicted_nvb)
    # print(X_test[:1])
    # print(twenty_train.target_names[y_test[1]])
    print("this predicted values")
    # print(twenty_train.target_names[predicted_svm[1]])
    print((text_clf_svm.predict_log_proba(newPredict)[0])[predicted_svm[0]])
    # print(text_clf_svm.predict_proba(newPredict[1],predicted_svm[1]))
    print(twenty_train.target_names[predicted_svm[0]])
    # print((text_clf_svm(newPredict)[0])[predicted_svm[0]])
    np.mean(predicted_svm == y_test)
    aList = []
    for i in range(len(predicted_svm)):
        pob =(text_clf_svm.predict_log_proba(newPredict)[i])[predicted_svm[i]]
        print(str(pob))
        print(twenty_train.target_names[predicted_svm[i]])
        aList.append({ "category" : str(twenty_train.target_names[predicted_svm[i]]),"pob": str(pob)})

    print(aList)
    return aList



#server
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

from io import BytesIO

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        try:
            body = json.loads(self.rfile.read(content_length))
            print(body['data'])
            cat = getCategories(body['data'])
            print(cat)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes(str(cat), "utf-8"))

        except Exception as e:
            print(str(e))
            self.send_response(401)
            self.end_headers()
            self.wfile.write({'message': str(e)})

httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
print(httpd.server_name+ httpd.server_port)

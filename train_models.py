from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef
base=Path(__file__).resolve().parent
out=base/'model';out.mkdir(parents=True,exist_ok=True)
d=load_breast_cancer(as_frame=True)
X=d.data.copy(); y=pd.Series(np.where(d.target==0,1,0),index=X.index,name='target')
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
full=X.copy(); full['target']=y.map({0:'benign',1:'malignant'});full.to_csv(base/'dataset.csv',index=False)
test=Xte.copy();test['target']=yte.map({0:'benign',1:'malignant'});test.to_csv(base/'test_data.csv',index=False)
models={'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42)), 'Decision Tree':DecisionTreeClassifier(max_depth=5,random_state=42),'KNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5)),'Naive Bayes':GaussianNB(),'Support Vector Machine':make_pipeline(StandardScaler(),SVC(kernel='rbf',probability=True,random_state=42))}
paths=['logistic_regression.joblib','decision_tree.joblib','knn.joblib','naive_bayes.joblib','svm.joblib']
rows=[]
for (name,clf),path in zip(models.items(),paths):
 clf.fit(Xtr,ytr)
 p=clf.predict(Xte)
 joblib.dump(clf,out/path)
 scores=clf.predict_proba(Xte)[:,list(clf.classes_).index(1)]
 rows.append(dict(Model=name,Accuracy=accuracy_score(yte,p),AUC=roc_auc_score(yte,scores),Precision=precision_score(yte,p),Recall=recall_score(yte,p),F1=f1_score(yte,p),MCC=matthews_corrcoef(yte,p)))
 print(name, 'Accuracy:',round(accuracy_score(yte,p),4),'F1:',round(f1_score(yte,p),4))
pd.DataFrame(rows).to_csv(base/'model_comparison.csv',index=False)
(base/'model_config.json').write_text(__import__('json').dumps({'feature_names':list(X.columns),'model_files':dict(zip(models,paths)),'positive_class':'malignant','test_rows':len(yte)},indent=2),encoding='utf-8')

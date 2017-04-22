from sklearn import cross_validation,svm,metrics

def load_csv(fname):
	labels=[]
	images=[]
	with open(fname,"r") as f:
		for line in f:
			cols=line.split(",")
			if len(cols)<2:continue
			labels.append(int(cols.pop(0)))
			vals=list(map(lambda n: int(n) / 256,cols))
			images.append(vals)
		return {"labels":labels,"images":images}

data=load_csv("./data/train.csv")
test=load_csv("./data/t10k.csv")

clf=svm.SVC()
clf.fit(data["images"],data["labels"])

predict=clf.predict(test["images"])

score=metrics.accuracy_score(test["labels"],predict)
report=metrics.classification_report(test["labels"],predict)
print(score)
print(report)
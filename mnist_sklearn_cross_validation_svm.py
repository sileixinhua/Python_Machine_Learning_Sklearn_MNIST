#用sklearn中的SVM来训练模型，预测数据集
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
#训练数据集

predict=clf.predict(test["images"])
#预测测试集

score=metrics.accuracy_score(test["labels"],predict)
#生成测试精度
report=metrics.classification_report(test["labels"],predict)
#生成交叉验证的报告
print(score)
#显示数据精度
print(report)
#显示交叉验证数据集报告
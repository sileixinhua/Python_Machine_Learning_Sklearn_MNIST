#将mnist数据集转换成CSV格式
import struct

def to_csv(name,maxdata):
	lbl_f = open("./data/"+name+"-labels.idx1-ubyte","rb")
	#打开标签数据集
	img_f = open("./data/"+name+"-images.idx3-ubyte","rb")
	#打开图像数据集
	csv_f = open("./data/"+name+",csv","w",encoding="utf-8")
	#写入CSV文件

	mag,lbl_count=struct.unpack(">II",lbl_f.read(8))
	#将字节流转换成python数据类型复制给标签
	mag,img_count=struct.unpack(">II",img_f.read(8))
	#将字节流转换成python数据类型复制给图像
	rows,cols=struct.unpack(">II",img_f.read(8))
	#将字节流转换成python数据类型复制给行列
	pixels=rows*cols
	#计算数据总量

	res=[]
	for idx in range(lbl_count):
		if idx > maxdata:break
		#设置计数器，大于数据个数总量时跳出循环
		label=struct.unpack("B",lbl_f.read(1))[0]
		bdata=img_f.read(pixels)
		sdata=list(map(lambda n:str(n),bdata))
		csv_f.write(str(label)+",")
		#写入标签
		csv_f.write(",".join(sdata)+"\r\n")
		#写入数据（数字）
		if idx < 10:
			s="P2 28 28 255\n"
			s+=" ".join(sdata)
			iname="./data/{0}-{1}-{2}.pgm".format(name,idx,label)
			with open(iname,"w",encoding="utf-8") as f:
				f.write(s)
	csv_f.close()
	#关闭CSV流
	lbl_f.close()
	#关闭标签流
	img_f.close()
	#关闭图像流

to_csv("train",1000)
#转换到train.csv 1000个数据
to_csv("t10k",1000)
#转换到t10k.csv 1000个数据
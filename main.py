from libsvm.svm import *
from libsvm.svmutil import *
from libsvm.commonutil import *

y, x = svm_read_problem('CM1_libsvm.txt')
options = '-c 1 -b 1'  # 选择训练参数
print("开始训练")
model = svm_train(y, x, options)  # 训练模型
svm_save_model('svm.model', model)  # 保存模型

# print("开始测试")
# yt, xt = svm_read_problem('C:/Users/hnt/Desktop/test.txt')  # 读入测试数据
# m = svm_load_model('svm.model')  # 读取模型
# p_label, p_acc, p_val = svm_predict(yt, xt, m, '-b 1')  # 预测
# ACC, MSE, SCC = evaluations(yt, p_label)

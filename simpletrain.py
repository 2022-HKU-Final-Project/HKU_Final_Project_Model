#!/usr/bin/env python
# coding: utf-8

# In[1]:


from simpletransformers.classification import ClassificationModel, ClassificationArgs,MultiLabelClassificationModel,MultiLabelClassificationArgs
import pandas as pd
import logging
import argparse
import numpy as np
from sklearn import preprocessing
 
 
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# In[2]:


data = pd.read_csv("./data/train_set_10w.csv")


# In[3]:


print(data.head())


# In[5]:


le = preprocessing.LabelEncoder()
le.fit(np.unique(data['true_job'].tolist()))
# print('标签值标准化:%s' % le.transform(["专业顾问", "美容师", "美发助理/学徒", "置业顾问","行政专员/助理"]))
# print('标准化标签值反转:%s' % le.inverse_transform([0, 80 ,79 ,78 ,81]))


# In[6]:


data['true_job'] = data['true_job'].apply(lambda x:le.transform([x])[0])


# In[7]:



from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(data['parsedText'], data['true_job'], test_size=.2,random_state=0,stratify=train_labels)


# In[8]:


train_dataset = pd.DataFrame(list(zip(train_texts,train_labels)))
val_dataset = pd.DataFrame(list(zip(val_texts,val_labels)))
# print(type(train_texts))


# In[4]:


print(data.jobCate.describe())


# In[ ]:


from sklearn.metrics import f1_score, accuracy_score
 

def f1_multiclass(labels, preds):
      return f1_score(labels, preds, average='micro')



if __name__ == '__main__':
# Optional model configuration
     parser = argparse.ArgumentParser()
     parser.add_argument('-m', help="name of model")
     model_args = ClassificationArgs(num_train_epochs=1000,
                                    evaluate_during_training=True,
                                    save_eval_checkpoints = False,
                                    save_model_every_epoch = True
                                    )
    args,_ = parser.parse_known_args()
    model_args.output_dir = args.m + "_outputs/"
    model_args.best_model_dir =  args.m + "_outputs/best_model"
#Create a ClassificationModel
     model = ClassificationModel(
     args.m,
    'distilbert-base-multilingual-cased',
     num_labels=65,
     args=model_args
    )

#Train the model
    model.train_model(train_dataset,eval_df = val_dataset)
 


# In[11]:


# load pretrained model
# from simpletransformers.model import TransformerModel
# new_model = BrandNewBertModel.load_pretrained_checkpoint('./outputs/checkpoint-50000')
# new_model = ClassificationModel('./outputs/checkpoint-50000')
# model = TransformerModel('distilbert', './outputs/best_model/', num_labels=1000)


# In[12]:


# result, model_outputs, wrong_predictions = model.eval_model(val_dataset)
# print(result)
# print(model_outputs)
# print(wrong_predictions)


# In[16]:


# test_text = ["岗位职责：负责公司人事方面资源的整理、整合；按照公司阶段性招聘计划利用智联、5赶集、前程无忧等大型招聘网站渠道，进行人才的信息收集、初步筛选、初步面试等；通过邀约，安排面试者到公司参加面试，及后期入职人员的跟进；健全公司人事制度、人员培训、人事资料管理；负责打印等方面。职位要求：大专以上学历，专业不限，欢迎2018年应届毕业生；具有较强的独立学习和工作的能力，工作踏实，认真细心，积极主动；具有良好的职业操守及团队合作精神，较强的沟通、理解和分析能力、福利待遇：无责底薪+奖金，均薪4000-5000元；入职起公司提供五险一金+双休+国家法定假日休息；每个季度公司组织去国内外旅游；过生的同事，公司会发生日礼品！！公司提供专业的免费的培训！享受降温费/取暖费 节日福利生日礼金等；每月15号工资准时到账，绝对不拖欠工资！",
#             "岗位职责：负则仓库日常工作 任职资格：任真负则工作时间：8小时有意者欢迎直接电话联系",
#              "岗位职责：负责子公司报表编制、账务处理及税务管理； 负责编制集团合并报表，编制并报送监管部门所要求的各类报表； 审计工作的沟通及协调； 上级交待的其他工作。岗位要求： 全日制大学本科以上； 英语读写能力较好； 注册会计师，熟悉国际会计准则，4年以上会计工作经验； 有审计经验。",
#              "岗位职责：详细了解家长对英语教育的理解和需求，并了解孩子的实际英语水平，帮助他们纠正错误的教育观念和教育方法，树立正确的英语教育、英语应用能力教育和人格教育的观念，帮助家长为孩子做好正确的人生抉择！在充分了解孩子的英语学习基础的前提下，和不同阶段的老师进行协商后，为孩子选择恰当的学习阶段。在家长决定购买意向后，督促家长认真阅读入学协议，帮助家长完成缴费的全部手续。时将手中的客户数据，按照公司的规范要求进行数据整理、录入和其他维护工作。及时提醒已经付费的家长，按照预定的日期，准时送孩子参加英语课程学习。以及其他一些必要的销售管理工作任职要求：您需要具备积极乐观的人生态度，您应该是一个相信“没有解决不了的问题，没有克服不了的困难”的人；您需要具备勤奋坚韧的性格，因为我们相信“天道酬勤，勤能补拙”；您需要具备尊重他人的良好修养；您还需要具备认真细致的工作作风；诚实正直并具有良好职业道德；",
#              "负责公司区块链产品的研发；公链，数字货交易所，DAPP，区块链底层区块链是代表未来的技术，观迎极客参与其中，改变世界负责公司区块链产品的研发；公链，数字货交易所，DAPP，区块链底层区块链是代表未来的技术，观迎极客参与其中，改变世界负责公司区块链产品的研发；公链，数字货交易所，DAPP，区块链底层区块链是代表未来的技术，观迎极客参与其中，改变世界区块链技术涉及面比较广，前后端我们都有细分的职位，我们都是需要的，主要后端的开发语言为 golang java 部份组件涉及c++，算法类，密码学，前端主要是php，ios 安卓因此，本职位是一个广泛职位，如你觉得技术上是比较有天分，涉及从事上述技术的知识及能力，不管前后端，都请尽快与我们联系根据需求开发项目。部分开发文档书写与整理。为公司快速开发框架积累自定义控件库。对自己开发的功能进行自测后提交测试部门测试。技术难点的研究和解决。职位要求：计算机科学或相关专业本以上学历；3年以上IOS开发经验；熟悉IOS应用开发框架，能独立开发高性能的IOS应用；精通Object-C或C/C++语言，具有Iphone、IPAD平台开发经验深入理解Objective-C Runtime运行机制和内存管理机制，深入了解各个不同iOS版本的特性与差异；精通IOS SDK 中的UI、熟悉多线程，网络编程（TCP/IP、HTTP）、XML/JSON解析等技术；对软件产品有强烈的责任心，良好的合作精神和工作态度，具有良好的创新心态者优先；数字贷币 区块链 blockchain 比特币 bitcoin 期贷现贷 股票 交易所 交易平台",
#              "岗位职责：参与项目需求分析,承担主要模块的设计开发工作；确定模块中难点实现的技术路线和设计，编写相应的说明书； 承担客户现场定制化开发和问题定位分析工作； 负责项目的资源调配与平衡，对项目进度的监控并协调解决过程产生的问题； 收集客户需求，定期与公司沟通，反馈客户产品新需求并整理成技术文档；带领小组人员完成相应软件模块的开发,培训带领员工参与开发模块控件。 任职要求： 本科及以上学历，计算机应用或相关专业； 三年以上软件开发实施、维护经验，熟悉银行、公安行业人员优先考虑； 具备良好的沟通能力，客户服务意识，具有高效的执行力、全局把控力； 熟悉windows 、linux、unix等操作系统和日常操作能力； 对oracle、dbsybase在unix、linux和windows等操作系统上的安装、部署能力； 熟练掌握java开发，并有项目开发经验； 熟悉tcp/ip通信， websphere mq通信优先。",
#              "",
#              "岗位职责：负责区块链底层相关模块的设计和开发； 负责区块链上层应用接口设计和开发； 根据公司要求进行的前瞻性实验原型的设计与开发；岗位要求： 3年以上Linux下开发经验，本科以上学历； 熟悉Golang/C/C++语言，精通Golang协程，对Golang的核心特性有深入了解，有过实际项目开发优化经验； 熟练掌握RESTful API规范，并有丰富开发经验； 熟练掌握多线程编程； 熟悉MySQL数据库、Redis等数据库； 有高并发分布式优化经验的优先考虑； 有安全相关领域开发经验或区块链项目开发经验者优先、 对计算机技术有强烈爱好和兴趣，主动沟通和协作，积极推进；邮箱地址：工作地点：北京市东城区雍和宫大象投资大厦",
#              "基本素质：学历：本科及以上 l 责任心：认真负责 l 思维水平：思维灵活 l 沟通能力、协调能力良好 l 学习能力强l 负责完成开发自测、发布、上线； l 负责完成独立模块的前后端实现； l 多个页面的HTML+CSS实现，及数据交互实现； l 协助完成文档撰写等；l 熟练使用mysql、oracle等至少一种关系数据库； l 熟练使用java开发语言； l 熟练使用tomcat、jetty、jboss等至少一种应用服务器； l 了解struts、spring、mybits、hibernate等框架； l 能熟练使用Git或SVN等项目工具； l 熟悉主流的SSH、SSM等应用开发框架； l 掌握HTML、JS、CSS基础； l 了解JQuery、Bootstrap等主流JavaScript库/框架；l 开发经验一年以上； l 至少负责过2个项目或或产品的开发，并有输出结果（如项目总结、工作手册案例等）；",
#             ]

# test_data = pd.read_csv('test.csv')
# test_data['jobCate'] = data['jobCate'].apply(lambda x:le.transform([x])[0])
# test_data = test_data['jobDesc'].values.tolist()
# result, model_outputs, wrong_predictions = model.eval_model(test_data)
# print(result)
# prediction,raw_outputs = model.predict(test_data)
# print(raw_outputs)
# print(le.inverse_transform(prediction))


# In[ ]:


# test_text =[
#     "负责分站区域内的新闻素材收集、实地采访、话题规划、热点挖掘并完成组稿、成稿任务；负责网站内容的信息搜集、编辑、优化、审校、更新，微信公众号运营等；协助政务发布宣传、推广的效果进行跟踪，及时提出改进措施。 任职要求： 30周岁以内，3-5年传统媒体记者岗位经验，时政新闻或电视台报社工作经验优先； 政治立场坚定，拥护党的各项方针政策，遵守宪法和法律，无犯罪记录； 获得过市级以上新闻类奖项者优先；全日制本科及以上学历，新闻类专业优先；面试需携带5篇以上出刊新闻作品；’特别优秀者薪资可面议。",
#     "岗位职责：通过手机接收客户订单； 严格按照操作流程，到指定地点餐厅进行取餐； 确保按时将客户的餐，送至制定地点；任职要求： 18-50周岁，对快递服务行业充满热情，能够适应快节奏的工作； 熟悉当地地形，有同行业工作经验者优先考虑； 熟练操作智能手机、电子地图，有较强的沟通能力，服从管理、正直、诚信、能吃苦耐劳； 有较强的服务意识，具有良好的团队精神； 年轻、有朝气、富有激情、性格开朗； 自备交通工具。福利待遇： 员工享有保护工资政策，薪资待遇属行业领先水平，确保员工福利。 公司统一为员工提供相应的保险。 公司将提供良好的课程培训，同时具有完善的员工晋升机制。工作时间：17：00---2：00",
#     "岗位要求：1年以上Web应用开发经验 对工作积极严谨，勇于承担压力 对业务、需求有较强的理解能力 有良好的沟通和学习能力、有强烈的团队协作意识和快速解决问题的能力; 有扎实的JAVA基础；熟悉主流的开源框架，熟练运用Spring MVC、Mybatis等 熟练使用JQuery、JavaScript、bootstrap、AJAX、echarts熟练掌握oracle或mysql数据库语法 有社保行业项目背景优先考虑全日制统招大专及以上学历1能适应出差"
# ]
# prediction,raw_outputs = model.predict(test_text)
# print(raw_outputs)
# print(le.inverse_transform(prediction))


# In[ ]:





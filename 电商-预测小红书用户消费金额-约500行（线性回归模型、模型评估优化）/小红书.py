#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#分析背景" data-toc-modified-id="分析背景-0.1">分析背景</a></span></li><li><span><a href="#分析目标" data-toc-modified-id="分析目标-0.2">分析目标</a></span></li><li><span><a href="#分析流程" data-toc-modified-id="分析流程-0.3">分析流程</a></span></li></ul></li><li><span><a href="#1-数据概况分析" data-toc-modified-id="1-数据概况分析-1">1 数据概况分析</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#指标解释" data-toc-modified-id="指标解释-1.0.1">指标解释</a></span></li></ul></li></ul></li><li><span><a href="#2-单变量分析" data-toc-modified-id="2-单变量分析-2">2 单变量分析</a></span><ul class="toc-item"><li><span><a href="#2.1-数字型变量" data-toc-modified-id="2.1-数字型变量-2.1">2.1 数字型变量</a></span></li><li><span><a href="#2.2-类别型变量" data-toc-modified-id="2.2-类别型变量-2.2">2.2 类别型变量</a></span></li></ul></li><li><span><a href="#3-相关可视化" data-toc-modified-id="3-相关可视化-3">3 相关可视化</a></span></li><li><span><a href="#4-回归模型" data-toc-modified-id="4-回归模型-4">4 回归模型</a></span><ul class="toc-item"><li><span><a href="#4.1-建立模型" data-toc-modified-id="4.1-建立模型-4.1">4.1 建立模型</a></span></li><li><span><a href="#4.2-模型评估" data-toc-modified-id="4.2-模型评估-4.2">4.2 模型评估</a></span></li><li><span><a href="#4.2-模型迭代优化" data-toc-modified-id="4.2-模型迭代优化-4.3">4.2 模型迭代优化</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#销售额模型预计分析：" data-toc-modified-id="销售额模型预计分析：-4.3.0.1">销售额模型预计分析：</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#5-业务建议" data-toc-modified-id="5-业务建议-5">5 业务建议</a></span><ul class="toc-item"><li><span><a href="#5.1-用户分析" data-toc-modified-id="5.1-用户分析-5.1">5.1 用户分析</a></span></li><li><span><a href="#5.2-提高销售额变量分析---高价值用户" data-toc-modified-id="5.2-提高销售额变量分析---高价值用户-5.2">5.2 提高销售额变量分析 - 高价值用户</a></span></li><li><span><a href="#5.3-结论" data-toc-modified-id="5.3-结论-5.3">5.3 结论</a></span></li></ul></li><li><span><a href="#6-模型解读" data-toc-modified-id="6-模型解读-6">6 模型解读</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#6.1-系数解读" data-toc-modified-id="6.1-系数解读-6.0.1">6.1 系数解读</a></span><ul class="toc-item"><li><span><a href="#销售额模型预计分析：" data-toc-modified-id="销售额模型预计分析：-6.0.1.1">销售额模型预计分析：</a></span></li></ul></li><li><span><a href="#6.2-业务解读" data-toc-modified-id="6.2-业务解读-6.0.2">6.2 业务解读</a></span></li></ul></li></ul></li></ul></div>

# ## 分析背景
# 说到小红书，是目前非常热门的电商平台，和其他电商平台不同，小红书是从社区起家。在小红书社区，用户通过文字、图片、视频笔记的分享，记录了这个时代年轻人的正能量和美好生活。
# 
# 小红书通过机器学习对海量信息和人进行精准、高效匹配，已累积海量的海外购物数据，分析出最受欢迎的商品及全球购物趋势，并在此基础上把全世界的好东西，以最短的路径、最简洁的方式提供给用户。
# 
# 本项目协助小红书分析不同的业务决策所带来的销售额变化。

# ## 分析目标
# 根据用户数据以及消费行为数据
# 
# - 使用Python建立线性回归模型
# - 预测用户的消费金额变化
# - 找到对用户消费影响较大的因素
# 

# ## 分析流程

# # 1 数据概况分析

# ### 指标解释
# - revenue：用户的下单购买金额
# - 3rd_party_stores：用户过往在app中从第三方商家购买的数量，0表示购买的自营产品
# - gender：男1，女0，空缺unkown
# - age：年龄，空缺unkown
# - engaged_last_30：最近30天在app上有参与重点活动
# - lifecycle：生命周期A：注册6个月内; B：一年内; C：两年内
# - days_since_last_order：最近一次下单距今的天数（小于1表示当天有下单）
# - previous_order_amount：累计的用户购买金额

# statsmodels包下载网站 https://pypi.org/project/statsmodels/#files 下载对应系统python版本安装包，项目目录有下载好的文件。
# 
# 安装参考指令：
# 
# pip install statsmodels-0.12.2-cp37-none-win_amd64.whl

# In[6]:


# 本人的python是3.9版本，所以运行下面指令安装
#get_ipython().system('pip install statsmodels-0.13.5-cp39-cp39-win_amd64.whl')


# In[40]:


#导入模块
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
get_ipython().run_line_magic('matplotlib', 'inline')

#导入数据
red = pd.read_csv('redNew.csv')
red.head()


# In[41]:


#查看数据信息
red.info()


# 结果显示共有29452行数据，其中gender, age, engaged_last_30三列有缺失值，需要对其进行数据处理

# In[42]:


#查看缺失值占总数的比值
red.isnull().sum()/red.count()


# 缺失值占比约为70%，不能直接删除缺失值，否则会损失非常多的数据

# In[43]:


#了解数据分布
red.describe()


# In[44]:


#对于连续变量，可以用均值、中位数或者根据其他数据模型填充；
#对于类别变量，则可以把变量拆解为哑变量，再删除重复或没有意义的变量
#gender是类别型变量，先把缺失值填充为unknown
#即将gender拆解为gender_0, gender_1, gender_unknown, 每个变量用0/1表示
red['gender'] = red['gender'].fillna('unknown')


# In[45]:


#engaged_last_30是类别型变量，也把缺失值填充为unknown
red['engaged_last_30'] = red['engaged_last_30'].fillna('unknown')


# In[46]:


#类别型变量填充完毕后，剩下的连续型变量均可用均值填充和进行哑变量处理
#age是连续变量，用均值填充
red = red.fillna(red.mean(numeric_only=True))
red.info()


# In[ ]:





# In[ ]:





# # 2 单变量分析

# ## 2.1 数字型变量

# In[47]:


#用户下单金额revenue
sns.distplot(red['revenue'])


# 集中分布在0-200之间

# In[48]:


#用户以往累积购买金额previous_order_amount
sns.distplot(red['previous_order_amount'])


# 用户以往累积金额为0-100的频数较高

# In[49]:


#年龄age
sns.distplot(red['age'])


# In[50]:


#年龄age的数据分布
red['age'].describe()


# - 用户年龄分布在18 - 99岁之间，用户年龄平均值为60.39岁，用户年龄中位数为60.39岁，绝大部分用户年龄集中在58 = 63岁之间
# - 用户年龄在平均值两侧呈对称分布
# - 29452个用户的年龄为60.3岁，60岁或是小红书的默认年龄选项

# In[51]:


#把 days_since_last_order 前面的空格删掉
red.rename(columns = {' days_since_last_order ':'days_since_last_order'}, inplace = True)
# red.info()

#用户最近一次下单距今的天数days_since_last_order
sns.distplot(red['days_since_last_order'])


# In[52]:


red['days_since_last_order'].describe()


# - 用户最近一次下单距今的天数分布在0 - 23天之间，可见所有的用户在本月内都有下单行为
# - 用户最近一次下单后平均7.7天后会再次下单
# - 用户最近一次下单距今0 - 1天的频数较高

# ## 2.2 类别型变量

# In[53]:


red


# In[54]:


#生命周期lifecycle
#不同生命周期(lifecycle)对应的revenue(销售额)是怎样的
#生命周期，分类为A,B,C（分别对应注册后6个月内，1年内，2年内）
red.groupby(red['lifecycle'])['revenue'].describe()


# In[55]:


#不同生命周期计数
#red['lifecycle'].value_counts(dropna = False).plot(kind = 'bar')
sns.countplot(x = 'lifecycle', data = red, order = red['lifecycle'].value_counts().index)


# In[56]:


#不同生命周期的销售额平均值
sns.barplot(x = 'lifecycle', y = 'revenue', data = red)


# In[57]:


#不同生命周期的销售额总和
sns.barplot(x = 'lifecycle', y = 'revenue', data = red, order = red['lifecycle'].value_counts().index, estimator = sum)


# In[58]:


red['lifecycle'].describe()


# - C(注册后两年)的用户最多
# - A(注册后6个月内的)用户销售额平均值最高，为433.8；其次是C(注册后两年)用户，销售额平均值为396.8；销售额平均值最低的是B(注册后一年内的)用户，销售额平均值为381.3
# - C用户销售额总和最高,其次是B用户,销售额总和最低的是A用户.
# - 总销售额大部分由C的用户创造
# 

# In[59]:


#不同性别(gender)对应的revenue(销售额)是怎样的
red.groupby(red['gender'])['revenue'].describe()


# In[60]:


#不同性别计数
sns.countplot(x = 'gender', data = red, order = red['gender'].value_counts().index)


# In[61]:


#不同性别的销售额平均值
#barplot() 默认展示的是某种变量分布的平均值（可通过参数修改为max、median 等）
sns.barplot(x = 'gender', y = 'revenue', data = red)


# In[62]:


#不同性别的销售额总和
sns.barplot(x = 'gender', y = 'revenue', data = red, estimator = sum)


# - 男性顾客远远超过女性顾客数量
# - 男性顾客的销售额平均值比女性顾客的消费额平均值略高
# - 总销售额大部分由男性顾客贡献

# In[63]:


#最近30天在APP上参与重要活动与否对应的销售额是怎样的
red.groupby(['engaged_last_30'])['revenue'].describe()


# In[64]:


#不同类别计数
sns.countplot(x = 'engaged_last_30', data = red, order = red['engaged_last_30'].value_counts().index)


# In[65]:


#最近30天在APP上参与重要活动与否对应的销售额平均值
sns.barplot(x = 'engaged_last_30', y = 'revenue', data = red)


# In[66]:


#最近30天在APP上参与重要活动与否对应的销售额总和
sns.barplot(x = 'engaged_last_30', y = 'revenue', data = red, estimator = sum)


# In[67]:


red['engaged_last_30'].describe()


# - 大部分用户30天内未在APP上参加活动
# - 最近30天在APP上参与了活动的用户的销售额平均值更大，为569.7，未参加活动的用户的销售额平均值为369.8元
# - 总销售额的大部分由30天内未在APP上参加活动的用户创造

# In[68]:


#用户过往在第三方APP购买的数量
red['3rd_party_stores'].describe()


# In[69]:


#用户过往在第三方APP购买的数量计数
sns.countplot(x = '3rd_party_stores', data = red, order = red['3rd_party_stores'].value_counts().index)


# In[70]:


#用户过往在第三方APP购买的数量对应的销售额平均值
sns.barplot(x = '3rd_party_stores', y = 'revenue', data = red)


# In[71]:


#用户过往在第三方APP购买的数量对应的销售额总和
sns.barplot(x = '3rd_party_stores', y = 'revenue', data = red, estimator = sum)


# - 从未在第三方APP购买过的顾客最多，其次是在第三方APP购买过10次的顾客
# - 从未在第三方APP购买过的顾客的平均销售额最多；在第三方APP进行了1 - 5次购买的顾客的平均销售额差不多；在第三方APP进行了6 - 10次购买的顾客的平均销售额差不多,不过后者大于在第三方APP进行了1 - 5次购买的顾客的平均销售额
# - 总销售额大部分由从未在第三方APP购买过的用户贡献，其次是在第三方APP购买过10次的用户，再次是在第三方APP进行了1 - 5次购买的顾客，贡献最少的是在第三方APP进行6 - 10次购买的顾客

# # 3 相关可视化

# In[72]:


red.info()


# gender, lifecycle, engaged_last_30均生成哑变量，days_since_last_order前面的空格也删除了

# In[73]:


# 将gender\lifecycle\engaged_last_30\生成哑变量
red=pd.get_dummies(red)
red.info()


# In[74]:


#cmap colormap(颜色映射)
sns.heatmap(red.corr(),cmap = 'Blues')


# - lifecycle_C和days_since_last_order和3rd_party_stores两两正相关关系较强
# - gender_1.0和engaged_last_30_0.0正相关关系较强
# - revenue和其他任何变量之间的相关性都不明显

# In[75]:


# 仅查看所有变量与revenue的相关性，同时根据相关性做降序排列展示
red.corr()[['revenue']].sort_values('revenue', ascending = False)


# - revenue和其他任何变量之间的相关性都不明显
# - 和revenue正相关性最高的是previous_order_amount, engaged_last_30_1.0, days_since_last_order
# - 和revenue负相关性最高的是3rd_party_stores, engaged_last_30_0.0, gender
# - 其他呈现弱相关关系的变量将不做解读，以免被过度解读
# 

# In[78]:


#对days_since_last_order变量进行线性关系可视化分析
sns.regplot(data=red,x='days_since_last_order',y='revenue')


# In[80]:


#对previous_order_amount变量进行线性关系可视化分析
sns.regplot(data=red,x='previous_order_amount',y='revenue')


# In[81]:


#对engaged_last_30_1.0变量进行线性关系可视化分析
sns.regplot(data=red,x='engaged_last_30_1.0',y='revenue')


# In[82]:


#对3rd_party_stores变量进行线性关系可视化分析
sns.regplot(data=red,x='3rd_party_stores',y='revenue')


# In[83]:


#对age变量进行线性关系可视化分析
sns.regplot(data=red,x='age',y='revenue')


# In[85]:


#对lifecycle_C变量进行线性关系可视化分析
sns.regplot(data=red,x='lifecycle_C',y='revenue')


# # 4 回归模型

# In[86]:


red.info()


# ## 4.1 建立模型

# In[87]:


#导入模块
from sklearn.linear_model import LinearRegression
#建立一个空回归模型
model = LinearRegression()
#设定自变量和因变量
y = red['revenue']
x = red[['previous_order_amount','engaged_last_30_1.0','days_since_last_order']]
#拟合
model.fit(x,y)
#查看系数
model.coef_


# In[88]:


#查看截距
model.intercept_


# ## 4.2 模型评估

# In[89]:


#给x和y打分
score = model.score(x,y)
#通过x计算y的预测值
predictions = model.predict(x)
#计算误差
error = predictions - y
#计算rmse
rmse = (error**2).mean()**.5
#计算mae
mae = abs(error).mean()

print(rmse)
print(mae)


# MAE（Mean Absolute Error）:
# - 绝对平均误差，是绝对误差的平均值
# - 把每个数据点的预测值和真实值相见，将所有数据点加总求平均
# - 可以更好地反映预测值误差的实际情况
# 
# RMSE（Root Mean Square Error）:
# - 均方根误差，将每个数据点的误差取平方后开方
# - RMSE比起MAE放大了误差，对误差的惩罚更重
# - 常用来作为机器学习模型预测结果衡量的标准

# In[90]:


from statsmodels.formula.api import ols
model = ols('y~x', red).fit()
print(model.summary())


# - R-squared为0.031，Prob为0
# - 该回归基本无意义

# ## 4.2 模型迭代优化

# In[91]:


#新增自变量lifecycle_C
#导入模块
from sklearn.linear_model import LinearRegression
#建立一个空回归模型
model = LinearRegression()
#设定自变量和因变量
y = red['revenue']
x = red[['previous_order_amount','engaged_last_30_1.0','days_since_last_order','lifecycle_C']]
#拟合
model.fit(x,y)


# In[92]:


#给x和y打分
score = model.score(x,y)
#通过x计算y的预测值
predictions = model.predict(x)
#计算误差
error = predictions - y
#计算rmse
rmse = (error**2).mean()**.5
#计算mae
mae = abs(error).mean()

print(rmse)
print(mae)


# In[93]:


from statsmodels.formula.api import ols
model = ols('y~x', red).fit()
print(model.summary())


# #### 销售额模型预计分析：
# 预计销售额 = 186.8375+0.0684*previous_order_amount+62.6521*engaged_last_30_1.0+8.9856*days_since_last_order-31.0249*lifecycle_C

# # 5 业务建议

# ## 5.1 用户分析

# - 用户年龄分布在18 - 99岁之间，用户年龄平均值为60.39岁，用户年龄中位数为60.39岁，绝大部分用户年龄集中在58 = 63岁之间
# - 用户年龄在平均值两侧呈对称分布
# - 29452个用户的年龄为60.3岁，60岁或是小红书的默认年龄选项

# ## 5.2 提高销售额变量分析 - 高价值用户

# - C(注册后两年)的用户最多
# - A(注册后6个月内的)用户销售额平均值最高，为433.8；其次是C(注册后两年)用户，销售额平均值为396.8；销售额平均值最低的是B(注册后一年内的)用户，销售额平均值为381.3
# - C用户销售额总和最高,其次是B用户,销售额总和最低的是A用户
# - 总销售额大部分由C的用户创造
# 
# 
# - 从未在第三方APP购买过的顾客最多，其次是在第三方APP购买过10次的顾客
# - 从未在第三方APP购买过的顾客的平均销售额最多；在第三方APP进行了1 - 5次购买的顾客的平均销售额差不多；在第三方APP进行了6 - 10次购买的顾客的平均销售额差不多,不过后者大于在第三方APP进行了1 - 5次购买的顾客的平均销售额
# - 总销售额大部分由从未在第三方APP购买过的用户贡献，其次是在第三方APP购买过10次的用户，再次是在第三方APP进行了1 - 5次购买的顾客，贡献最少的是在第三方APP进行6 - 10次购买的顾客
# 
# 
# - lifecycle_C和days_since_last_order和3rd_party_stores两两正相关关系较强
# - gender_1.0和engaged_last_30_0.0正相关关系较强
# - revenue和其他任何变量之间的相关性都不明显

# ## 5.3 结论

# - 重点留意注册后两年用户的留存情况
# - 活动、广告、营销重点针对男性用户
# - 鼓励用户在APP上参加活动，特别是男性用户
# - 针对从未在第三方APP进行过购买活动的用户进行额外推送、广告、营销、优惠券推送等，其次是在第三方APP进行购买活动大于等于10次的用户

# # 6 模型解读

# ### 6.1 系数解读

# #### 销售额模型预计分析：
# 预计销售额 = 186.8375+0.0684*previous_order_amount+62.6521*engaged_last_30_1.0+8.9856*days_since_last_order-31.0249*lifecycle_C
# 
# - previous_order_amount：用户以往累计购买金额
# - engaged_last_30_1.0：用户在最近30天内参加过了活动：1，未参加活动：0
# - days_since_last_order：用户最近一次下单距今的天数
# - lifecycle_C：用户是在两年前注册：1，不是在两年前注册：0

# ### 6.2 业务解读

# - 每提升1元的用户以往累积购买金额，可以得到0.06元的销售额回报
# - 用户用户在最近30天内参加过活动比起不参加活动，销售额提升了62.65元
# - 用户最近一次下单距今的天数每提升1天，则可以实现8.98元的销售回报
# - 用户在两年前注册比起不在两年前注册，销售额降低了31.02元
# - 不断收集数据和添加新变量能提升对整体营销资源投入的把握
# （注：该数据线性回归效果较差，仅为练习线性回归步骤用）

#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#项目介绍" data-toc-modified-id="项目介绍-1">项目介绍</a></span><ul class="toc-item"><li><span><a href="#关键词" data-toc-modified-id="关键词-1.1">关键词</a></span></li></ul></li><li><span><a href="#一、读取数据" data-toc-modified-id="一、读取数据-2">一、读取数据</a></span></li><li><span><a href="#二、数据清洗" data-toc-modified-id="二、数据清洗-3">二、数据清洗</a></span><ul class="toc-item"><li><span><a href="#2.1-重复数据处理" data-toc-modified-id="2.1-重复数据处理-3.1">2.1 重复数据处理</a></span></li><li><span><a href="#2.2-缺失值处理" data-toc-modified-id="2.2-缺失值处理-3.2">2.2 缺失值处理</a></span></li><li><span><a href="#2.3-提取表格中有用信息并新增为列" data-toc-modified-id="2.3-提取表格中有用信息并新增为列-3.3">2.3 提取表格中有用信息并新增为列</a></span></li></ul></li><li><span><a href="#三、数据探索" data-toc-modified-id="三、数据探索-4">三、数据探索</a></span><ul class="toc-item"><li><span><a href="#3.1-各品牌SKU数" data-toc-modified-id="3.1-各品牌SKU数-4.1">3.1 各品牌SKU数</a></span></li><li><span><a href="#3.2-品牌总销量和总销售额" data-toc-modified-id="3.2-品牌总销量和总销售额-4.2">3.2 品牌总销量和总销售额</a></span></li><li><span><a href="#3.3-各类别的销售量、销售额情况" data-toc-modified-id="3.3-各类别的销售量、销售额情况-4.3">3.3 各类别的销售量、销售额情况</a></span></li><li><span><a href="#3.4-各品牌热度" data-toc-modified-id="3.4-各品牌热度-4.4">3.4 各品牌热度</a></span></li><li><span><a href="#3.5-各品牌价格" data-toc-modified-id="3.5-各品牌价格-4.5">3.5 各品牌价格</a></span></li><li><span><a href="#3.6-男性护肤品销量情况" data-toc-modified-id="3.6-男性护肤品销量情况-4.6">3.6 男性护肤品销量情况</a></span></li><li><span><a href="#3.7-分析时间与销量的关系，体现购买高峰期" data-toc-modified-id="3.7-分析时间与销量的关系，体现购买高峰期-4.7">3.7 分析时间与销量的关系，体现购买高峰期</a></span></li></ul></li><li><span><a href="#总结" data-toc-modified-id="总结-5">总结</a></span></li><li><span><a href="#建议" data-toc-modified-id="建议-6">建议</a></span></li></ul></div>

# ## 项目介绍
# 
# 数据为天猫双十一女性美妆的数据集，围绕产品及其销量和评论撰写。数据具有7个特征，可以从多个维度解析文本。
# 由于是真实的商业数据，所以做了匿名处理，数据集中对店名的引用被处理为产品的品牌名以保护店家隐私。
# 
# 数据包括27599行和7个特征变量。每一行对应一个产品的销售情况，包括以下变量:
# 
# update_time	统计时间
# 
# id    产品编号
# 
# title    产品名称
# 
# price    交易价格
# 
# sale_count    销量
# 
# comment_count    评论数量
# 
# 店名    	店铺名称
# 
# 可参考的探索方向：
# 
# 购买化妆品的客户的关注度(评论数)是多少？各产品销量分布情况？
# 哪些产品的卖得最好，哪些牌子最受欢迎，哪些化妆品是大家最需要的？
# 不同商家之间的差异，及促销打折力度？
# 模拟定价系统及推荐系统？

# ###  关键词
# 
# - pandas数据处理、matplotlib、数据可视化、数据探索分析

# ## 一、读取数据

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_csv('双十一淘宝美妆数据.csv')
data.head()


# In[10]:


# 查看各字段信息
data.info()


# In[11]:


# 分店铺统计
data['店名'].value_counts()


# ## 二、数据清洗

# ### 2.1 重复数据处理

# In[12]:


# 对重复数据做删除处理
print(data.shape)
data = data.drop_duplicates(inplace=False)
print(data.shape)


# In[13]:


# 此处虽然删除了重复值，但索引未变，因此应用以下方法进行重置索引
print(data.index)
data = data.reset_index(drop=True)
print('新索引：',data.index)


# ### 2.2 缺失值处理

# In[15]:


# 查看缺失值
data.isnull().any()


# 有两列数据存在缺失值：sale_count, comment_count

# In[16]:


# 查看数据结构
data.describe()


# In[17]:


# 查看sale_count列的众数
mode_01 = data.sale_count.mode()
print(mode_01)

# 查看comment_count列的众数
mode_02 = data.comment_count.mode()
print(mode_02)


# 此处两列的众数均为0，且由标签数据含义可知销售量和评论数有可能存在为0的情况，因此使用0来填充缺失值

# In[19]:


# 填充缺失值
data = data.fillna(0)
# 对空值行数求和
data.isnull().sum()


# ### 2.3 提取表格中有用信息并新增为列

# In[20]:


# 结巴分词库
import jieba 
# jieba.load_userdict('addwords.txt')
title_cut = []
for i in data.title:
    j = jieba.lcut(i)
    title_cut.append(j)
    
# 对标题进行分词，新增item_name_cut列
data['item_name_cut'] = title_cut
data[['title','item_name_cut']].head()


# In[32]:


# 给商品添加分类
basic_config_data = """护肤品    套装    套装                            
护肤品    乳液类    乳液    美白乳    润肤乳    凝乳    柔肤液'    亮肤乳    菁华乳    修护乳
护肤品    眼部护理    眼霜    眼部精华    眼膜                    
护肤品    面膜类    面膜                                                    
护肤品    清洁类    洗面    洁面    清洁    卸妆    洁颜    洗颜    去角质    磨砂                        
护肤品    化妆水    化妆水    爽肤水    柔肤水    补水露    凝露    柔肤液    精粹水    亮肤水    润肤水    保湿水    菁华水    保湿喷雾    舒缓喷雾
护肤品    面霜类    面霜    日霜    晚霜    柔肤霜    滋润霜    保湿霜    凝霜    日间霜    晚间霜    乳霜    修护霜    亮肤霜    底霜    菁华霜
护肤品    精华类    精华液    精华水    精华露    精华素                                        
护肤品    防晒类    防晒霜    防晒喷雾                                                
化妆品    口红类    唇釉    口红    唇彩                                            
化妆品    底妆类    散粉    蜜粉    粉底液    定妆粉     气垫    粉饼    BB    CC    遮瑕    粉霜    粉底膏    粉底霜        
化妆品    眼部彩妆    眉粉    染眉膏    眼线    眼影    睫毛膏                                    
化妆品    修容类    鼻影    修容粉    高光    腮红                                        
其他    其他    其他"""

# 将字符串basic_config_data 转为字典 category_config_map
category_config_map = {}
for config_line in basic_config_data.split('\n'):
    basic_cateogry_list = config_line.strip().strip('\n').strip('    ').split('    ')
    main_category = basic_cateogry_list[0]
    sub_category = basic_cateogry_list[1]
    unit_category_list = basic_cateogry_list[2:-1]
    for unit_category in unit_category_list:
        if unit_category and unit_category.strip().strip('    '):
            category_config_map[unit_category] = (main_category,sub_category)
            
category_config_map


# In[33]:


def func1(row):
    sub_type = ''   #子类别
    main_type = '' #主类别
    exist = False
    # 遍历item_name_cut 里每个词语
    for temp in row:
        # 如果词语包含在category_config_map里面，打上子类和主类标签
        if temp in category_config_map:
            sub_type = category_config_map.get(temp)[1]
            main_type = category_config_map.get(temp)[0]
            exist = True
            break
    if not exist:
        sub_type= '其他'
        main_type = '其他'
        
    return [sub_type, main_type]


# In[34]:


# 将子类别sub_type新增为一列
data['sub_type'] = data['item_name_cut'].map(lambda r:func1(r)[0])
# 将主类别main-type新增为一列
data['main_type'] = data['item_name_cut'].map(lambda r:func1(r)[1])
data.head()


# In[35]:


# 将“是否男士专用”新增为一列
gender = []
for i in range(len(data)):
    if '男' in data.item_name_cut[i]:
        gender.append('是')
    elif '男士' in data.item_name_cut[i]:
        gender.append('是')
    elif '男生' in data.item_name_cut[i]:
        gender.append('是')
    else:
        gender.append('否')
        
# 将“是否男士专用”新增为一列
data['是否男士专用'] = gender
data.head()


# In[37]:


# 新增销售额、购买日期（天）为一列
# 销售额=销售量*价格
data['销售额'] = data.sale_count*data.price

# 新增购买日期为一列
# 转换时间格式
data['update_time'] = pd.to_datetime(data['update_time'])

data[['update_time']].head()


# In[38]:


# 将时间设置为新的index
data = data.set_index('update_time')

# 新增时间“天”为一列
data['day'] = data.index.day

# 删除中文分词的一列
del data['item_name_cut']


# In[40]:


# 查看最终数据表格
data.head()


# In[41]:


# 保存清理好的数据为Excel格式
data.to_excel('./clean_beautymakeup.xlsx',sheet_name='clean_data')


# ## 三、数据探索

# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data.columns


# ### 3.1 各品牌SKU数

# In[47]:


plt.rcParams['font.sans-serif']=['SimHei']  #指定默认字体  
plt.rcParams['axes.unicode_minus']=False  #解决负号'-'显示为方块的问题

plt.figure(figsize=(12,7))
# 计算各店铺的商品数量
data['店名'].value_counts().sort_values(ascending=False).plot.bar(width=0.8,alpha=0.6,color='b')

plt.title('各品牌SKU数',fontsize=18)
plt.ylabel('商品数量',fontsize=14)
plt.show()


# 悦诗风吟的商品数量最多，其次为佰草集、欧莱雅

# ### 3.2 品牌总销量和总销售额

# In[48]:


fig,axes = plt.subplots(1,2,figsize=(12,10))

ax1 = data.groupby('店名').sale_count.sum().sort_values(ascending=True).plot(kind='barh',ax=axes[0],width=0.6)
ax1.set_title('品牌总销售量',fontsize=12)
ax1.set_xlabel('总销售量')

ax2 = data.groupby('店名')['销售额'].sum().sort_values(ascending=True).plot(kind='barh',ax=axes[1],width=0.6)
ax2.set_title('品牌总销售额',fontsize=12)
ax2.set_xlabel('总销售额')

plt.subplots_adjust(wspace=0.4)
plt.show()


# 相宜本草的销售量和销售额都是最高的。销量第二至第五，分别为美宝莲、悦诗风吟、妮维雅、欧莱雅；销售额第二至第五，分别为欧莱雅、佰草集、美宝莲、悦诗风吟。 宝莲、悦诗风吟、欧莱雅都在销量、销售额前五中。

# ### 3.3 各类别的销售量、销售额情况

# In[87]:


fig,axes = plt.subplots(1,2,figsize=(18,12))

data1 = data.groupby('main_type')['sale_count'].sum()
ax1 = data1.plot(kind='pie',ax=axes[0],autopct='%.1f%%', # 设置百分比的格式，这里保留一位小数
pctdistance=0.8, # 设置百分比标签与圆心的距离
labels= data1.index,
labeldistance = 1.05, # 设置标签与圆心的距离
startangle = 60, # 设置饼图的初始角度
radius = 1.2, # 设置饼图的半径
counterclock = False, # 是否逆时针，这里设置为顺时针方向
wedgeprops = {'linewidth': 1.2, },# 设置饼图内外边界的属性值
textprops = {'fontsize':16, 'color':'k','rotation':80}, # 设置文本标签的属性值
)
ax1.set_title('主类别销售量占比',fontsize=20)

data2 = data.groupby('sub_type')['sale_count'].sum()
ax2 = data2.plot(kind='pie',ax=axes[1],autopct='%.1f%%', 
pctdistance=0.8, 
labels= data2.index,
labeldistance = 1.05,
startangle = 230, 
radius = 1.2, 
counterclock = False, 
wedgeprops = {'linewidth': 1.2, },
textprops = {'fontsize':16, 'color':'k','rotation':80}, 
)

ax2.set_title('子类别销售量占比',fontsize=20)

# 设置坐标标签
ax1.set_xlabel(..., fontsize=16,labelpad=38.5)
ax1.set_ylabel(..., fontsize=16,labelpad=38.5)
ax2.set_xlabel(..., fontsize=16,labelpad=38.5)
ax2.set_ylabel(..., fontsize=16,labelpad=38.5)
plt.subplots_adjust(wspace=0.4)
plt.show()


# 从主类别销售量占比情况来看，护肤品的销量远大于化妆品；
# 
# 从子类别销售量占比情况来看，底妆类、口红类在化妆品中销量最多，清洁类、化妆水、面霜类在护肤品中销量最多。

# In[101]:


plt.figure(figsize=(18,8))
sns.barplot(x='店名',y='sale_count',hue='main_type',data=data,saturation=0.75,ci=0)
plt.title('各品牌各总类的总销量', fontsize=20)
plt.ylabel('销量',fontsize=16)
plt.xlabel('店名',fontsize=16)
plt.text(0,78000,'注：此处也可使用堆叠图，对比效果更直观',
         verticalalignment='top', horizontalalignment='left',color='gray', fontsize=10)
# 设置刻度字体大小

plt.xticks(fontsize=16,rotation=45)
plt.yticks(fontsize=16)
plt.show()


# In[105]:


plt.figure(figsize=(18,8))
sns.barplot( x = '店名',
y = '销售额',hue = 'main_type',data =data,saturation = 0.75,ci=0,)
plt.ylabel('销售额',fontsize=16)
plt.xlabel('店名',fontsize=16)
plt.title('各品牌各总类的总销售额',fontsize=20)
# 设置刻度字体大小
plt.xticks(fontsize=16,rotation=45)
plt.yticks(fontsize=16)
plt.show()


# 各品牌的化妆品、护肤品销量、销售情况均不一样，这与品牌的定位有关， 有的品牌主打化妆品，化妆品会表现好很多，如蜜丝佛陀等。主打护肤品的品牌，护肤品的销量销售额会表现好很多，如欧莱雅、佰草集等。 有的品牌如美宝莲、兰蔻、悦诗风吟，化妆品和护肤品的销售、销售额都还不错。

# In[107]:


plt.figure(figsize = (16,6))
sns.barplot( x = '店名',
y = 'sale_count',hue = 'sub_type',data =data,saturation = 0.75,ci=0)
plt.title('各品牌各子类的总销量')
plt.ylabel('销量')
plt.show()


# In[106]:


plt.figure(figsize = (14,6))
sns.barplot( x = '店名',
y = '销售额',hue = 'sub_type',data =data,saturation = 0.75,ci=0)
plt.title('各品牌各子类的总销售额')
plt.ylabel('销售额')
plt.show()


# ### 3.4 各品牌热度

# In[108]:


plt.figure(figsize = (12,6))
data.groupby('店名').comment_count.mean().sort_values(ascending=False).plot(kind='bar',width=0.8)
plt.title('各品牌商品的平均评论数')
plt.ylabel('评论数')
plt.show()


# In[111]:


plt.figure(figsize=(18,12))

x = data.groupby('店名')['sale_count'].mean()
y = data.groupby('店名')['comment_count'].mean()
s = data.groupby('店名')['price'].mean()
txt = data.groupby('店名').id.count().index

sns.scatterplot(x,y,size=s,hue=s,sizes=(100,1500),data=data)

for i in range(len(txt)):
    plt.annotate(txt[i],xy=(x[i],y[i]))
    
plt.ylabel('热度')
plt.xlabel('销量')

plt.legend(loc='upper left')
plt.show()


# 由上图所示：
# 
# 越靠上的品牌热度越高，越靠右的品牌销量越高，颜色越深圈越大价格越高
# 
# 热度与销量呈现一定的正相关；
# 美宝莲热度第一，销量第二，妮维雅热度第二，销量第四，两者价格均相对较低；
# 价格低的品牌热度和销量相对较高，价格高的品牌热度和销量相对较低，说明价格在热度和销量中有一定影响；

# ### 3.5 各品牌价格

# In[113]:


#查看价格的箱型图
plt.figure(figsize=(18,10))
sns.boxplot(x='店名',y='price',data=data)
plt.ylim(0,3000)#如果不限制，就不容易看清箱型，所以把Y轴刻度缩小为0-3000
plt.show()


# In[114]:


data.groupby('店名').price.sum()
avg_price=data.groupby('店名').price.sum()/data.groupby('店名').price.count()
avg_price


# In[117]:


fig = plt.figure(figsize=(12,6))
avg_price.sort_values(ascending=False).plot(kind='bar',width=0.8,alpha=0.6,color='b',label='各品牌平均价格')
y = data['price'].mean()
plt.axhline(y,0,5,color='r',label='全品牌平均价格')
plt.ylabel('各品牌平均价格')
plt.title('各品牌产品的平均价格',fontsize=24)
plt.legend(loc='best')
plt.show()


# 娇兰、SKII、雪花秀、雅诗兰黛、兰蔻、资生堂这几个国际大牌的产品价格很高，产品平均价格都在500以上，都是一线大牌；
# 
# 兰芝、倩碧、玉兰油、植村秀、佰草集、薇姿、雅漾的平均价格在300-400元左右，其中佰草集是最贵的国货品牌；
# 
# 美加净作为国货品牌，性价比高，平均价格最低，妮维雅的平均价格第二低，在100元左右；
# 
# 全品牌平均价格低于400元，除了前五个国际大牌其余品牌的平均价格都低于全品牌平均价格；
# 

# In[120]:


plt.figure(figsize=(18,10))

x = data.groupby('店名')['sale_count'].mean()
y = data.groupby('店名')['销售额'].mean()
s = avg_price
txt = data.groupby('店名').id.count().index

sns.scatterplot(x,y,size=s,sizes=(100,1500),marker='v',alpha=0.5,color='b',data=data)

for i in range(len(txt)):
    plt.annotate(txt[i],xy=(x[i],y[i]),xytext = (x[i]+0.2, y[i]+0.2))  #在散点后面增加品牌信息的标签
    
plt.ylabel('销售额')
plt.xlabel('销量')

plt.legend(loc='upper left')
plt.show()


# 由上图所示，越靠上代表销售额越高，越靠左代表销量越高，图形越大代表平均价格越高
# 
# 销售量和销售额呈现正相关；
# 
# 相宜本草、美宝莲、蜜丝佛陀销量和销售额位居前三，且平均价格居中；
# 
# 说明销量销售额与价格有很重要的联系；

# ### 3.6 男性护肤品销量情况

# In[123]:


gender_data=data[data['是否男士专用']=='是']
gender_data_1=gender_data[(gender_data.main_type =='护肤品')| (gender_data.main_type=='化妆品')]
plt.figure(figsize = (12,6))
sns.barplot(x='店名',y='sale_count',hue='main_type',data =gender_data_1,saturation=0.75,ci=0,)
plt.show()


# In[129]:


f,[ax1,ax2]=plt.subplots(1,2,figsize=(12,6))
gender_data.groupby('店名').sale_count.sum().sort_values(ascending=True).plot(kind='barh',width=0.8,ax=ax1)
ax1.set_title('男士护肤品销量排名')

gender_data.groupby('店名').销售额.sum().sort_values(ascending=True).plot(kind='barh',width=0.8,ax=ax2)
ax2.set_title('男士护肤品销售额排名')

plt.subplots_adjust(wspace=0.4)
plt.show()


# 男士购买的大多是护肤品；
# 
# 妮维雅是男生护肤品中销量遥遥领先的品牌，第二第三分别为欧莱雅、相宜本草；

# ### 3.7 分析时间与销量的关系，体现购买高峰期

# In[132]:


from matplotlib.pyplot import MultipleLocator
plt.figure(figsize = (12,6))
day_sale=data.groupby('day')['sale_count'].sum()
day_sale.plot()
plt.grid(linestyle="-.",color="gray",axis="x",alpha=0.5)
x_major_locator=MultipleLocator(1)  #把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()  #ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
plt.xlabel('日期（11月）',fontsize=12)
plt.ylabel('销量',fontsize=12)
plt.show()


# ## 总结

# 美妆类别中护肤品销量远大于化妆品，其中清洁类、化妆水、面霜等基础护肤类销量最高；
# 
# 男士购买美妆集中在护肤品类，其中妮维雅品牌是最受男士喜爱的品牌；
# 
# 价格和热度对销售量有关联，平价基础产品是大多数消费者的选择；
# 
# 由于商家在双十一提前预热，巨大的优惠力度和为了避免网络高峰，不少消费者选择提前消费，销量高峰出现在双十一前几天；双十一后3天商家持续打折优惠，消费者还保有购物余热，但远不如双十一之前。

# ## 建议

# 消费者对产品价格和热度关注度较高，品牌可以适当调整产品价格并通过诸如网络社交平台的方式提高品牌热度；
# 
# 对于男性消费者，品牌可以定向推荐平价基础护肤产品，在销量中可以看到也有一部分男性购买化妆品，品牌可以在护肤品中适当捆绑化妆品产品带动消费；
# 
# 消费者购买欲望并不集中在双十一当天，商家可以提前预热加大优惠力度刺激消费者提前消费，避免网络高峰。

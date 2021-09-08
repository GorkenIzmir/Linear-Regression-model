#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
df=pd.read_csv("ormanyangin.csv")


# In[51]:


##df.drop('month',axis=1,inplace=True)
##df.drop('day',axis=1,inplace=True)   yapıldı
df


# In[52]:


#1. Montesinho park haritasındaki X - x ekseni uzamsal koordinatı: 1 ila 9
#2. Y - y ekseni Montesinho park haritası içindeki uzamsal koordinat: 2 ila 9
#3. yılın ayı: 'jan' ila ' aralık'
#4. gün - haftanın günü: 'mon' ila 'sun'
#5. FFMC - FWI sisteminden FFMC indeksi: 18.7 - 96.20
#6. DMC - FWI sisteminden DMC indeksi: 1.1 - 291.3
#7. DC - FWI sisteminden DC indeksi: 7,9 ila 860,6
#8. FWI sisteminden ISI - ISI indeksi: 0,0 ila 56,10
#9. temp sıcaklık - Santigrat derece cinsinden sıcaklık: 2,2 ila 33,30
#10. RH - % olarak bağıl nem: 15,0 ila 100
#11. wind-rüzgar - km/sa cinsinden rüzgar hızı: 0,40 - 9,40
#12. (rain)yağmur - aa / m2 dışında yağmur: 0.0 6.4
#13. area - (ha) orman yanmış alanı: 0.00 1090,84 için    ha=hektar=10,000 m2

#tekli reg
#hangi rüzgar hızında  ne kadar alan yandı tahmini


# In[53]:


import seaborn as sns
sns.jointplot(x="wind",y="area", data=df,kind="reg")


# In[54]:


x=df[["wind"]]
y=df[["area"]]
from sklearn.linear_model import LinearRegression


# In[55]:


reg=LinearRegression()


# In[56]:


model=reg.fit(x,y)


# In[57]:


model.intercept_  # doğrusal b0 katsayısı


# In[58]:


model.coef_ # doğrusal b1 katsayısı


# In[59]:


import matplotlib.pyplot as plt


# In[60]:


g=sns.regplot(df["wind"],df["area"],ci=None,scatter_kws={'color':'r','s':9})
g.set_ylabel("area")
g.set_xlabel("rüzgar hızı")
plt.xlim(0,20) #limitler
plt.ylim(0,50)


# In[61]:


#doğrusal reg tahmin
model.predict([["8"]])


# In[62]:


#çoklu reg


# In[65]:


X=df.drop('area',axis=1)
Y=df[["area"]]


# In[66]:


X


# In[67]:


Y


# In[68]:


from sklearn.linear_model import LinearRegression
l=LinearRegression()


# In[69]:


model2=l.fit(X,Y)


# In[70]:


model2.intercept_


# In[71]:


model2.coef_


# In[72]:


tah=[["4"],["6"],["80"],["75"],["600"],["12"],["20"],["40"],["10"],["0"]]
tah=pd.DataFrame(tah).T
tah


# In[73]:


model2.predict(tah)


# In[74]:


## sınama


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.20,random_state=99)


# In[77]:


model3=l.fit(x_train,y_train)


# In[82]:


import numpy as np
from sklearn.metrics import mean_squared_error
train_error=np.sqrt(mean_squared_error(y_train,model3.predict(x_train))) #hata kareler top
train_error


# In[83]:


test_error=np.sqrt(mean_squared_error(y_test,model3.predict(x_test)))
test_error


# In[84]:


from sklearn.model_selection import cross_val_score


# In[85]:


cross_val_score(model3,x_train,y_train,cv=10,scoring="neg_mean_squared_error")


# In[86]:


np.sqrt(np.mean(-cross_val_score(model3,x_train,y_train,cv=10,scoring="neg_mean_squared_error")))


# In[89]:


#ridge regresyon ile model
##gerekli kütüpler
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV


# In[143]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[144]:


ridge_model=Ridge(alpha=3).fit(x_train,y_train)


# In[145]:


ridge_model


# In[146]:


ridge_model.intercept_


# In[147]:


ridge_model.coef_


# In[155]:


lambdalar=10**np.linspace(10,-2,100)-0.5


# In[156]:


katsayilar=[]


# In[157]:


for i in lambdalar:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(x_train,y_train)
    katsayilar.append(ridge_model.coef_)


# In[158]:


katsayilar


# In[159]:


lambdalar


# In[160]:


x


# In[164]:


type(katsayilar)


# In[165]:


#ax=plt.gca()
#ax.plot(lambdalar,katsayilar)
#ax.set_xscale("log")
#çalıştıramadım


# In[ ]:


#tahmin ksmı


# In[167]:


#train hatası 
Y_pred=ridge_model.predict(x_train)
RMSE=np.sqrt(mean_squared_error(y_train,Y_pred))


# In[168]:


RMSE


# In[169]:


#test hatası
Y_predtest=ridge_model.predict(x_test)
RMSE=np.sqrt(mean_squared_error(y_test,Y_predtest))
RMSE


# In[170]:


#model tuning


# In[171]:


lambdalar1=np.random.randint(0,1000,100)


# In[173]:


ridgecv=RidgeCV(alphas=lambdalar1,scoring="neg_mean_squared_error",cv=10,normalize=True)
ridgecv.fit(x_train,y_train)


# In[175]:


ridge_tuned=Ridge(alpha=ridgecv.alpha_).fit(x_train,y_train)


# In[177]:


y_predtest=ridge_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_predtest))###bizim 
#test hatamız 97.46263612177405 çıkmış idi burada test hatamız 97,46... basamaktan sonra yükselmiştir.modelimizi doğrulamış olduk


# In[ ]:





# In[ ]:





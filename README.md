# Cross-Domain GAN For Face Super-Resolution
## 應用  
### 監控系統  
低解析度還原高解析度技術，增強圖像的品質和細節可見性  
### 人臉識別系統  
低解析度人臉圖像還原高解析度，可以提高識別系統的性能  
      
## 困難  
### 重建上的困難:  
超解析是一個上採樣的問題，通常情況下，存在著多個低解析度樣本，這些樣本在外觀上相似，但對應的高解析度呈現顯著的樣貌差異。這樣的對應關係往往極為複雜。  
### 識別上的困難:  
解析度較低時，人臉的鑑別度下降，使得識別變得複雜。而當提高解析度時，雖然人臉的鑑別度增加，卻同時面臨著資料不足的問題，容易導致模型過度擬合。  

## 研究  
### Regression model  
加深網路模型，在回歸模型中加入殘差式學習以及漸進式的學習，每一個階段皆導入損失函數，使得每一個階段的預估值與正確答案一致，傳統方法下，通常都在最後階段才限制預估值與正確答案一致，我們的作法可以使得模型更加穩定。  
**本研究利用回歸模型有效解決在實務應用上時常存在著多個低解析度樣本，這些樣本在外觀上相似，但對應的高解析度呈現顯著的樣貌差異。這樣的對應關係往往極為複雜，但透過回歸模型能有效使得有著相同高解析的低解析圖像之間在編碼空間中具有聚類性。**   

### 跨域生成模型  
利用低解析編碼與對應高解析編碼之間的對應關係且兩者共享相同的編碼器(Encoder)，建構低解析域以及高解析域之間的相互關聯性。  

我們的架構圖如下圖所示  
##### 架構圖  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/3cd14ed9-4663-4bd3-97df-23d48f8e189a)


##### 回歸分布圖  
![]()  

##### ablation study(有無加入回歸的限制)  
![]()  



### 超解析識別  
提出了一個跨解析度的分類器，其中低解析度分類模型具有較強的一般性，但缺乏鑑別性，相對而言，高解析度分類模型則更具鑑別性，但容易因資料不足而產生過度擬合的現象。最終，我們結合了這兩者，以彌補各自的缺點。 

#### 識別模型  
![]()  

## GAN Inversion

#### 演算法
![]()  

#### 結果圖  
![]()  

### 識別結果圖  
**![]() 








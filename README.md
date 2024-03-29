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

**資料來源以及前處理如下圖所示**  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/4c2d811b-803d-4812-a0d7-03bc3bbdf582)  


我們的架構圖如下圖所示  
#### 架構圖  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/3cd14ed9-4663-4bd3-97df-23d48f8e189a)  

#### 損失函數  
            我們將利用這些損失函數，修正我們模型。
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/df17bf41-7389-42ed-baab-72d0649faff7)  

#### 訓練程序  
            (1)	訓練上半部壓縮還原模型(Enc,Dec)  
            (2)	訓練回歸模型(Reg)  
            (3)	訓練生成對抗網路(Gan)  
            (4)	端到端的整體轉換模型訓練(End to End)  
            (5)	跨域的分類模型訓練(teacher, student model)  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/4f4d4d44-08cf-42e2-bd73-bedafef3b476)  


#### 回歸分布圖  
            左側展示了未經過回歸模型處理的空間分布，而右側展示了經過回歸模型處理後的空間分布。顯然可見，經過回歸模型處理後，每個身分不同解析度之間呈現顯著聚合現象。  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/ee13ce8d-f5e1-4783-930b-1b7d473bbe9d)


#### ablation study(有無加入回歸的限制)  
            評估在訓練過程中是否添加𝐋𝐫𝐞𝐠對於測試資料的視覺化和量化結果有何影響。   
            (a) 高解析圖像  
            (b) 由左至右分別為4倍、6.4倍以及8倍低解析圖像，6.4倍率為模型未學習過倍率  
            (c) 𝐿_𝑟𝑒𝑐^𝑑+𝐿_𝑟𝑒𝑐^𝑔+𝐿_(𝐺_𝑎𝑑𝑣 )+𝐿_(𝐷_𝑎𝑑𝑣 )  
            (d) 𝐿_𝑟𝑒𝑐^𝑑+𝐿_𝑟𝑒𝑐^𝑔+𝐿_(𝐺_𝑎𝑑𝑣 )+𝐿_(𝐷_𝑎𝑑𝑣 )+𝐿_𝑟𝑒𝑔  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/e3e176aa-522e-4d20-9529-a5a0c7653560)



### 超解析識別  
提出了一個跨解析度的分類器，其中低解析度分類模型具有較強的一般性，但缺乏鑑別性，相對而言，高解析度分類模型則更具鑑別性，但容易因資料不足而產生過度擬合的現象。最終，我們結合了這兩者，以彌補各自的缺點。 

#### 識別模型  
1. 學生模型的輸入為各種不同的低解析圖像經過編碼器(Enc)產生的編碼Z_(i,j,r)^ 。老師模型的輸入為真實高解析圖像經過編碼器(Enc)產生的編碼Z_(i,j,r=1)^ 。並經由回歸模型（Reg）產生相應的編碼Z_(i,j,r)^( 〖reg〗^3 )。
2. 最後我們將Student模型和Teacher模型輸出相加，學生模型學習多樣低解析度資料，鑑別率不足但資料多；老師模型學習高解析度，鑑別率高但資料少容易過擬合，相加彌補彼此缺點，最終識別實驗證明這樣的概念可以有效的提高準確率。

#### GAN Inversion  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/3e0f78ab-b6e8-4dc4-b8af-f40c363bb929)
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/ca07facc-f666-43f3-b72a-eac7f515eefc)  

#### 演算法  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/a482066b-b955-4dcb-9f6a-fc08a7a0a766)  


#### 合成結果圖  
驗證在不同倍率以及不同模糊下不同方法下的合成結果。  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/fef150cc-0bb5-4177-95e3-05911ec23a0b)  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/5c836371-7919-4ce1-9ff0-98daa2d06b0b)  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/b5573f44-8f13-4bcc-ae55-964470079c51)  


#### 量化數據  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/51eb3fb5-5dd2-4390-b77f-c729cfd1e32a)  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/dabadb5f-39a6-4225-a807-5fbc5fa2df61)  

### 識別結果圖  
1. 提出的Student和Teacher輸出相加，有效提高準確率。  
2. 在沒學過的倍率且解析度較低的情況下，透過inversion 限制teacher的分類結果要跟student 分類結果一致，來提高teacher 的效能，進而讓整體識別提高  
![image](https://github.com/wangbosen123/Cross-Domain-GAN-for-Face-Super-Resolution/assets/92494937/c5356323-5f8e-4741-b91b-3324e4992b07)









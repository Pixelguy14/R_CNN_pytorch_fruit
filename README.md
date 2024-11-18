# R_CNN_pytorch_fruit
Region-based and Image-based Convolutional Neural Network made to identify fruits from images using pyTorch. Image dataset thanks to<br>
Seth, K. (2019). Fruit and Vegetable Image Recognition Dataset. Kaggle. Retrieved from https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition <br>
Filo, C. (2017). Fruit Recognition Dataset. Kaggle. Retrieved from https://www.kaggle.com/datasets/chrisfilo/fruit-recognition/data <br>
KOKLU, M., KURSUN, R., TASPINAR, Y. S. and CINAR, I. (2021). Classification of Date Fruits into Genetic Varieties Using Image Analysis. Mathematical Problems in Engineering, Vol.2021, Article ID: 4793293. from https://www.muratkoklu.com/datasets/ <br>
El-Chimminut, A. (2021). Fruits 262 Dataset. Kaggle. Retrieved from https://www.kaggle.com/datasets/aelchimminut/fruits262 <br>


## to run object detection CNN:
if you have a csv of data, run convert_csv_to_voc.py<br>
then execute trainVOC.py to train the Pytorch R CNN<br>
to test the model, run inference.py<br>
## for image detection CNN:
run imageClassifier.py<br>
to test the model, run image_inference.py<br>

# DhivehiNewsClassifier
Simple Keras model for classifying Dhivehi news topics based on article content. Inpired by [this article](https://becominghuman.ai/news-topic-classification-using-lstm-a1e8a38781fe) published on becominghuman.ai.

## Setup
- Create a virtual environment (recommended)
- ```git clone https://github.com/ashraq1455/dhivehi-news-classifier.git```
- ```pip install -r requirements.txt```
- Run ```gdown --id 1aYSqY7jY7MAK_j1t0hRTf6R7QjvO_u1I``` to download the dataset (18.5 MB)
- Extract and move ```dataset.csv``` to ```data``` directory

#### Dataset Structure:
|category|title_body|
|----|----|
|politics|ރައީސް ޔާމީންގެ މެސެޖެއް: އަލުން ޕީޕީއެމްއަށް ރާއްޖޭގެ އިގްތިސާދު ސަލާމަތް ކުރެވޭނެ!. އަނެއްކާވެސް އަލުން ޕީޕީއެމްއަށް ރާއްޖޭގެ އިގްތިސާދު ސަލާމަތްކޮށް، ޝަރުއީ އަދި ގާނޫނީ ދާއިރާ އަށް މާނަވީ އިސްލާހްތައް ގެނެވޭނެ ކަމަށް ކުރީގެ ރައީސް، އަދި އިދިކޮޅު ކޯލިޝަންގެ ލީޑަރު އަބްދުﷲ ޔާމީން އަބްދުލް ގައްޔޫމް ވިދާޅުވެއްޖެއެވެ
|sport|މެސީގެ ހާއްސަ ޓްރިބިއުޓެއް މަރަޑޯނާއަށް. ދާދިފަހުން މަރުވި އާޖެންޓީނާގެ ލެޖެންޑް ޑިއެގޯ މަރަޑޯނާގެ ހަނދާނުގައި ބާސާއަށް ކުޅެމުން އަންނަ ތަރި ލިއޮނަލް މެސީ ހާއްސަ ޓްރިބިއުޓެއް ދީފިއެވެ. ސްޕެނިޝް ލީގުގައި އޮސަސޫނާ އަތުން 4-0 އިން ބާސެލޯނާ މޮޅުވި މި މެޗުގައި މެސީ ޓްރިބިއުޓް އެއް ދީފައިވަނީ އޭނާ މެޗުގެ ހަތަރުވަނަ ގޯލު ޖެހުމަށްފަހު އެވެ|

## Train
- After setup run ```python train.py```. 

## Inference
- To load the most recently trained model run ```python inference.py```
- To load the model with best accuracy run ```python inference.py --best```
- To load a specific model run ```python inference.py --model``` and select a model

## Hyperparameters Tuning
To search best hyperparameters for your dataset, specify ranges for ```output_dim```, ```hp_learning_rate```, ```input_unit``` and run ```python search_best_hparams.py```.

## Pre-trained Models


|name|file_id|tra_accuracy|val_accuracy|size|
|----|----|----|----|----|
|mvnews_classifier_1622907065|11Bm_h9PXKYyWlv3Rw_2vHMcNES0Zl1CT|99.75%|95.74%|87.5 MB|

- To Download pre-trained models run ```gdown --id <file_id>```
- Extract the file and move all content to ```models``` directory
- To run inference on the model run ```python inference.py```

This model is trained using 33068 articles scrapped from vaguthu.mv. It can currently predict 5 categories: ```Politics, Business, Sports, World-News, Report```

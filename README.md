# SSL_Ensemble

## Project Struture

project-root-directory/
<br>
│
├── data/                  
│   ├── tfrecord_files       
│   └── tfindex_files          
│
├── weights/               
│   ├── checkpoint1.pt        
│   └── checkpoint2.pt                 
│
├── Myloader.py
|
├── model.py
|
├── train.ipynb
|
├── SimCLR_data.py
|
├── ssl_encoder.py
|
├── SimCLR.ipynb
|
├── ssl_pretrain.ipynb
|
└── counting.ipynb

## Usage
Run train.ipynb for ResNet18 backbone or ssl_pretrain.ipynb for SimCLR backbone.
The evaluation function for calculating average precision per-class is in train.ipynb.
The file counting.ipynb shows the data distribution for the X-ray dataset.

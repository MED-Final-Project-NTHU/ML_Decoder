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
├── Myloader.py <br>
|
├── model.py <br>
|
├── train.ipynb <br>
|
├── SimCLR_data.py <br>
|
├── ssl_encoder.py <br>
|
├── SimCLR.ipynb <br>
|
├── ssl_pretrain.ipynb <br>
|
└── counting.ipynb <br>

## Usage
Run train.ipynb for ResNet18 backbone or ssl_pretrain.ipynb for SimCLR backbone.
The evaluation function for calculating average precision per-class is in train.ipynb.
The file counting.ipynb shows the data distribution for the X-ray dataset.

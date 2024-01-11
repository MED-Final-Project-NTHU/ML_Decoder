# ML Decoder

## Root Folder - ML decoder

### Usage
1. Install the required packages by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

2. Downloading Data: Before running the ML decoder, you need to download the required data. 

- Download the image folder from [Download Link](insert_link_here) and save it to the `data/` folder.
   
- Download the CSV file from [Download Link](insert_link_here) and also save it to the `data/` folder.

- Your 'data' folder structure should look like this:

    - **data/**
        - **(Place the downloaded image files here)**
        - **(Place the downloaded CSV file here)**

3. To train the ConvNet + ML decoder network, run the following command:

```bash
python run_tran.py
```

(Note: You can change the XRay_Dataset to the original dataset format)

---

## Folder ssl - SSL_Ensemble

### Usage
1. For ResNet18 backbone, open and run the `train.ipynb` notebook.

2. For SimCLR backbone, open and run the `ssl_pretrain.ipynb` notebook.

3. To calculate the average precision per-class during evaluation, refer to the evaluation function in the `train.ipynb` notebook.

4. To visualize the data distribution for the X-ray dataset, you can use the `counting.ipynb` notebook.

---

Feel free to explore and use the provided scripts and notebooks in this repository for your machine learning decoder project. If you encounter any issues or have questions, please don't hesitate to reach out.
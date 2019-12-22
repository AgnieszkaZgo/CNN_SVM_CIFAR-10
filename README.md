# Inception and Support Vector Machine on CIFAR-10 dataset




Repository contains:

1. Main part of work, which will guide you through the whole learning and analysis process in a jupyter notebook file:   **Analysis_cifar-10.ipynb**  
   
2. Files with python code to download dataset and extract features using InceptionV3 network:   
**extract_features.py**  
**flip_extract_features.py**  
Extracted features are saved to: *codes_train.pkl*, *codes_train_flip.pkl*, *codes_test.pkl* in */home/agnieszka/* directory.

3. Files with python code to find best Support Vector Classifier on CNN codes extracted by foregoing scripts:  
**cifar-inc.py**  
**cifar-inc-st.py**  
**cifar-inc-st-aug.py**  
**cifar-inc-st-bagg.py**  
Best models found by GridSearchCV are saved to: *best_model.sav*, *best_model_st.sav*, *best_model_st_aug.sav*, *best_model_st_bagg.sav* in */home/agnieszka/* directory.

4. All the files saved during the work are available at:  
https://drive.google.com/file/d/16DABubDSfuapGQRcq0quPxovDUEu0Pjb/view?usp=sharing

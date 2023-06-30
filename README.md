# Condensed-Nearest-Neighbors

Custom Condensed Nearest Neighbors algorithm implementation, compatible with Scikit-learn class **KNeighborsTransformer** and with **PyNNDescentTransformer**, hence can be used in machine learning related projects, similarily as any other Transformer class.


The implementation and the subsets' terminology is based on: **Hart, Peter. "The condensed nearest neighbor rule (corresp.)."** IEEE transactions on information theory 14.3 (1968): 515-516. [link](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7c3771fd6829630cf450af853df728ecd8da4ab2)


It serves essentially as a **training set reduction** algorithm.


Tested on the ["Codon usage"] bioinformatics dataset, presented in article [Open Access version link](https://www.nature.com/articles/s41598-023-28965-7).


## Results

- Sampleset reduction in function of CNN steps
<img src="https://github.com/basiav/Condensed-Nearest-Neighbors/blob/main/output/output.png"/>


- Statistics:
```
-------------------- CNN SAFETY CHECK --------------------

CNN prediction on training data is the same as on     original KNN before sample reduction
y_pred_cnn_train == y_train).all():     True
CNN score on training dataset: 1.0
CNN score on reduced dim dataset: 1.0
CNN f1 score for training dataset: 1.0

-------------------- TIMES --------------------

KNN fit time: 0.00
CNN fit time: 0.00
CNN actual training (transform) time: 111.28

KNN prediction time: 0.00
CNN prediction time: 0.05

-------------------- ACCURACY --------------------

KNN F1 score: 89.16%
CNN F1: 86.13%
Samples before: 10406, samples after: 1296
% of sampleset reduction: 87.55

Process finished with exit code 0
```

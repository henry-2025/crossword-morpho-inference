import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay



predictions = trainer.predict(test_dataset).label_ids
most_common=[8, 9, 17, 11, 5]
paired_test, paired_predictions = [], []
for i in range(len(test_labels)):
    if test_labels[i] in most_common:
        paired_test.append(most_common.index(test_labels[i]))
        if predictions[i] in most_common:
            paired_predictions.append(most_common.index(predictions[i]))
        else:
            paired_predictions.append(5)
        
unique_labels = ['NN', 'NNP', 'VB', 'NNS', 'JJ', 'other']

fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay.from_predictions(paired_test, paired_predictions, normalize='all', labels=list(np.arange(6)), display_labels=unique_labels, include_values=False, ax=ax)
plt.savefig('test', dpi=300)
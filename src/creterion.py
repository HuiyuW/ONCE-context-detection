import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import random
import matplotlib as mpl


class ConfusionMatrix(object):


    def __init__(self, num_classes: int):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = [0,1,2,3]

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):#
        # 
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]#
        acc = sum_TP / n#
        # 
		
		# 
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # 
        kappa = round((po - pe) / (1 - pe), 3)
        #
        
        #
        table = PrettyTable()#
        table.field_names = ["", "Precision", "Recall", "Specificity","F1"]
        for i in range(self.num_classes):#
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.#
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2*TP / (2*TP + FN + FP), 3) if 2*TP + FN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity,F1])
        # 
        return str(acc),table


    def plot(self):#
        matrix = self.matrix
        acc,table = self.summary()
        #
        fig = plt.figure(num = random.randint(7,40))
        plt.imshow(matrix, cmap=plt.cm.Blues)
        cmap=plt.cm.Blues
        norm = mpl.colors.Normalize(vmin=0,vmax=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # 
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 
        plt.yticks(range(self.num_classes), self.labels)
        # 
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+acc+')')

        #
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                #
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        #

        pic_name = './results/accuracy' + '_' + acc +'.png'
        plt.savefig(pic_name,bbox_inches='tight')

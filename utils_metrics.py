import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pandas.plotting import table


# count the number of images in each class in the dataset
def count_images_in_dataset(dataset_dir):
    # list the sub folders in the dataset folder with the folders names being the class names
    classes = os.listdir(dataset_dir)

    # create a dictionary with the class names as keys and the number of images in each class as values
    class_images = {}

    for class_ in classes:
        class_dir = os.path.join(dataset_dir, class_)
        if os.path.isdir(class_dir):
            class_images[class_] = len(os.listdir(class_dir))
        else:
            print(f'Error: {class_dir} is not a directory')

    return class_images
    

# plot the dataset distribution for the training and test datasets
def bar_plot_dataset_distribution(train_images_dir, test_images_dir,save_path):
    # count the number of images in each class in the training and test datasets
    train_class_images = count_images_in_dataset(train_images_dir)
    test_class_images = count_images_in_dataset(test_images_dir)

    plt.figure(figsize=(10, 5))
    plt.bar(train_class_images.keys(), train_class_images.values(), width=0.5, color='g')
    plt.bar(test_class_images.keys(), test_class_images.values(), width=0.5, color='b')
    plt.xticks(rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.title('Number of images in each class')

    # add count labels to the bars
    for i, v in enumerate(train_class_images.values()):
        plt.text(i - 0.25, v + 10, str(v))
    
    # compute the total number of images in the training and test datasets
    total_train_images = sum(train_class_images.values())
    total_test_images = sum(test_class_images.values())

    # show legend
    plt.legend(['Train Images: ' + str(total_train_images), 
                'Test Images: ' + str(total_test_images)])

    # save the plot
    plt.savefig(save_path, bbox_inches='tight')

    # return the total number of images in the training and test datasets as a tuple
    return total_train_images, total_test_images


# plot a pie chart of the class distribution in the dataset for the training and test datasets as subplots
def pie_plot_dataset_distribution(train_images_dir, test_images_dir, save_path):

    # count the number of images in each class in the training and test datasets
    train_class_images = count_images_in_dataset(train_images_dir)
    test_class_images = count_images_in_dataset(test_images_dir)

    # plot a pie chart
    plt.figure(figsize=(15, 15))
    
    plt.subplot(1, 2, 1)
    plt.rcParams.update({'font.size': 11})
    plt.pie(train_class_images.values(), labels = train_class_images.values(), autopct='%.1f%%')
    # increase the title font size
    plt.rcParams.update({'font.size': 20})
    plt.title('Training Dataset')

    plt.subplot(1, 2, 2)
    plt.rcParams.update({'font.size': 11})
    plt.pie(test_class_images.values(), labels=test_class_images.values(), autopct='%.1f%%')
    # increase the title font size
    plt.rcParams.update({'font.size': 20})
    
    plt.title('Testing Dataset')

    plt.rcParams.update({'font.size': 11})

    # add legend and centre it at the middle of the two subplots at the bottom
    plt.legend(test_class_images.keys(), loc='lower center', bbox_to_anchor=(-0.1, -0.1), ncol=len(test_class_images.keys()), fancybox=True, shadow=True)

    # save the plot
    plt.savefig(save_path, bbox_inches='tight')


# get the unique labels with the true and predicted labels from the probability table
def get_true_and_pred_labels(prob_Path):
    # load the csv file
    df = pd.read_csv(prob_Path)
    # get the GroundTruth and Predicted labels columns using the names of the columns
    df = df[['GroundTruth', 'ModelPrediction']]
    # convert the dataframe to a list of lists
    true_and_pred_labels = df.values.tolist()
    # get the true and predicted labels
    true_labels = [i[0] for i in true_and_pred_labels]
    predicted_labels = [i[1] for i in true_and_pred_labels]
    # get the unique labels
    labels = list(set(true_labels))
    return true_labels, predicted_labels, labels


# define a function to plot the confusion matrix
def plot_confusion_matrix(prob_Path, save_path):
    # get the true and predicted labels and the unique labels
    true_labels, predicted_labels, labels = get_true_and_pred_labels(prob_Path)

    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    # sort the labels alphabetically
    sorted_labels = sorted(labels)

    # create a new confusion matrix with the sorted labels
    sorted_cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(sorted_labels):
        for j, pred_label in enumerate(sorted_labels):
            sorted_cm[i, j] = cm[labels.index(true_label), labels.index(pred_label)]

    # plot the confusion matrix as a heatmap including the count values
    fig = plt.figure(figsize=(7, 6))

    # plot the heatmap using seaborn, avoid displaying zero values
    sns.heatmap(sorted_cm, cmap='Blues', annot=True, fmt='d', cbar=False, linewidth=1, linecolor='white', mask=(sorted_cm == 0))

    plt.xlabel("Predicted labels", fontsize=12)
    plt.ylabel("True labels", fontsize=12)

    # set the tick positions and labels
    tick_positions = np.arange(len(sorted_labels)) + 0.5
    plt.xticks(tick_positions, sorted_labels, fontsize=10, rotation=40, ha='center')
    plt.yticks(tick_positions, sorted_labels, fontsize=10, rotation=0, va='center')

    # centre the labels in middle of each square
    plt.tick_params(axis='both', which='both', length=0, pad=10)

    plt.title('Confusion matrix', fontsize=14)

    # reduce the whitespace between the plot and the title
    plt.tight_layout()

    # adjust the padding of the plot to make the bottom visible
    fig.subplots_adjust(bottom=0.2)
    
    # save the plot
    fig.savefig(save_path, bbox_inches='tight')


# plot the recall, precision and f1 scores anr return the accuracy
def plot_rpf1(prob_Path, save_path):

    # get the true and predicted labels and the unique labels
    true_labels, predicted_labels, labels = get_true_and_pred_labels(prob_Path)
    # get the classification report
    report = classification_report(true_labels, predicted_labels, labels=labels, output_dict=True)

    # create a DataFrame from the classification report
    df = pd.DataFrame(report).transpose()
    # order the DataFrame by the labels
    df = df.sort_index()
    
    # get the accuracy from the classification report
    accuracy = df.loc['accuracy', 'f1-score']

    # create a new figure
    fig = plt.figure()    

    # plot the DataFrame as a heatmap
    sns.heatmap(df[['precision', 'recall', 'f1-score']], annot=True, cmap='Blues', fmt='.2f',linewidths=0.2, cbar=True, annot_kws={"size": 10})
    # decrease font size of values in the heatmap
    plt.tick_params(axis='both', which='both', labelsize=9)
    
    # adjust the padding of the plot to make the bottom visible
    plt.subplots_adjust(bottom=0.1)
    # adjust the padding of the plot to make the left visible
    plt.subplots_adjust(left=0.2)
    # adjust the padding of the plot to make the top visible
    plt.subplots_adjust(top=0.95)   

    # save the plot
    fig.savefig(save_path, bbox_inches='tight')

    return accuracy


# define a function to plot the the confidence of the correct predictions
def plot_cc_confidence(prob_Path, save_path):
    # get the probability table
    df = pd.read_csv(prob_Path)

    # TODO: change next line so it uses the column names instead of column indices
    df = df[df.iloc[:, 1] == df.iloc[:, 2]]

    # get the first column as ground truth labels
     # TODO: change next line so it uses the column names instead of column indices
    true_labels = df.iloc[:, 1]

    # get the probability values from the rest of the columns
     # TODO: change next line so it uses the column names instead of column indices
    probabilities = df.iloc[:, 3:]

    # get max values from each row in the probability table
    max_probabilities = probabilities.max(axis=1)

    # get labels
    labels = list(set(true_labels))

    # plot a boxplot of the max probabilities for each class
    plt.figure(figsize=(10, 10))
    meanprops = dict(marker='o', markeredgecolor='green', markerfacecolor='green')
    boxprops = dict(linestyle='-', linewidth=1, color='darkblue', facecolor='lightblue')
    # set the marker shape of the outliers to a diamond
    flierprops = dict(marker='.', markerfacecolor='black', markersize=5)

    # add notch, median and mean to the boxplot
    sns.boxplot(x=true_labels, y=max_probabilities, order=sorted(labels), flierprops=flierprops, boxprops= boxprops, showmeans=True, meanprops=meanprops)

    # plot the y axis between 0 and 1.1
    plt.ylim(0, 1.01)

    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.title('Probability for correct predictions on each class')
    
    # save the plot
    plt.savefig(save_path, bbox_inches='tight')


# define a function to plot the the confidence of the incorrect predictions
def plot_ic_confidence(prob_Path, save_path):

    # get the probability table
    df = pd.read_csv(prob_Path)
     # TODO: change next line so it uses the column names instead of column indices
    df = df[df.iloc[:, 1] != df.iloc[:, 2]]

    # get the first column as ground truth labels
     # TODO: change next line so it uses the column names instead of column indices
    true_labels = df.iloc[:, 1]

    # get the probability values from the rest of the columns
     # TODO: change next line so it uses the column names instead of column indices
    probabilities = df.iloc[:, 3:]

    # get max values from each row in the probability table
    max_probabilities = probabilities.max(axis=1)

    # get labels
    labels = list(set(true_labels))

    # plot a boxplot of the max probabilities for each class
    plt.figure(figsize=(10, 10))
    meanprops = dict(marker='o', markeredgecolor='green', markerfacecolor='green')
    boxprops = dict(linestyle='-', linewidth=1, color='darkblue', facecolor='lightblue')
    # set the marker shape of the outliers to a diamond
    flierprops = dict(marker='.', markerfacecolor='black', markersize=5)

    # add notch, median and mean to the boxplot
    sns.boxplot(x=true_labels, y=max_probabilities, order=sorted(labels), flierprops=flierprops, boxprops= boxprops, showmeans=True, meanprops=meanprops)

    # plot the y axis between 0 and 1.1
    plt.ylim(0, 1.01)

    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.title('Probability for incorrect predictions on each class')
    
    # save the plot
    plt.savefig(save_path, bbox_inches='tight')


# define a function to plot the misclassification cost matrix
def plot_misclassification_cost_matrix(prob_Path, RiskScorePlot_SavePath, costMatrixPath, pfmeaTable_SavePath):

    # read the csv file
    df = pd.read_csv(costMatrixPath, index_col='Labels', header=0)

    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, 2))  # Adjust the size

    # Remove axes
    ax.axis('tight')
    ax.axis('off')

    # Reset the index
    df_reset = df.reset_index()

    # Create the table and assign it to a variable
    table = ax.table(cellText=df_reset.values, colLabels=df_reset.columns, loc='center')

    # Center the values in the cells
    for key, cell in table.get_celld().items():
        cell.set_text_props(horizontalalignment='center')

    # Adjust table scale
    table.auto_set_font_size(True)

    table.set_fontsize(14)  # Set font size to 12

    # Increase the width of the columns
    table.scale(1.5, 1.5)  # Increase the first parameter as needed

    # Save pfmea table as figure
    plt.savefig(pfmeaTable_SavePath, dpi=1000, bbox_inches='tight')

    # get only values from the dataframe
    costMatrixValues = df.values

    # get the true and predicted labels and the unique labels
    true_labels, predicted_labels, labels = get_true_and_pred_labels(prob_Path)

    # get the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    # sort the labels alphabetically
    sorted_labels = sorted(labels)

    # create a new confusion matrix with the sorted labels
    sorted_cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(sorted_labels):
        for j, pred_label in enumerate(sorted_labels):
            sorted_cm[i, j] = cm[labels.index(true_label), labels.index(pred_label)]

    # normalize the sorted confusion matrix (normalization is done by dividing each row by the sum of the row)
    normalized_cm = sorted_cm.astype('float') / sorted_cm.sum(axis=1)[:, np.newaxis]
    # fix the decimal places to 2
    normalized_cm = np.around(normalized_cm, decimals=3)

    # check if costMatrixValues and normalized_cm have the same shape
    if costMatrixValues.shape != normalized_cm.shape:
        print('Error: The shape of the cost matrix and the normalized confusion matrix are not the same')
        # save and empty plot
        # create an emty figure
        plt.figure(figsize=(10, 5))
        plt.savefig(RiskScorePlot_SavePath, bbox_inches='tight')
        riskScore = 'Unknown'
        return riskScore

    # perform a multiplication of the normalized confusion matrix with the cost matrix and sum each row
    riskScore = np.sum(np.multiply(normalized_cm, costMatrixValues), axis=1)

    # add the labels to the weighted values with ascending order
    riskScore = dict(zip(sorted_labels, riskScore))

    # fix the decimal places to 2
    riskScore = {k: round(v, 2) for k, v in riskScore.items()}

    # compute the total risk score
    total_risk_score = round(sum(riskScore.values()), 2)

    # add the total risk score as a text on top right of the plot with font size 15
    # plt.text(5, 3.2, 'Total Risk Score: ' + str(total_risk_score), fontsize=15)

    # plot the weighted values as a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(riskScore.keys(), riskScore.values(), color='lightblue')
    # inlude the weighted values on top of each bar
    for i, v in enumerate(riskScore.values()):
        plt.text(i - 0.2, v + 0.01, str(v), color='black')

    plt.xlabel('Labels')
    plt.ylabel('Risk Score')
    plt.title('Weighted risk score for each class')
    # add total total_risk_score in the title
    plt.title('Weighted risk score for each class\n(Total Risk Score: ' + str(total_risk_score)+')')

    # make the risk score values in red if they are greater a threshold
    for i, v in enumerate(riskScore.values()):
        if v > 2:
            plt.text(i - 0.2, v + 0.01, str(v), color='red')


    # make x axis labels 45 degrees
    plt.xticks(rotation=45)

    # save the plot
    plt.savefig(RiskScorePlot_SavePath, bbox_inches='tight')

    return total_risk_score





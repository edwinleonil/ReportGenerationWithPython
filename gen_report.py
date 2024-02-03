from docxtpl import DocxTemplate, InlineImage
from docxtpl import InlineImage
from docx.shared import Mm, Inches, Pt
import logging
import datetime
import pandas as pd
from docx2pdf import convert
import utils_metrics
import yaml
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the Word document template
template = DocxTemplate('report_template.docx')

# get date time
now = datetime.datetime.now()

# convert to uk format
date = now.strftime("%d/%m/%Y")

# load the paths from the YAML file
with open('def_file_paths.yml', 'r') as f:
    paths = yaml.safe_load(f)

# access the paths using the dictionary keys
training_dataPath = paths['training_dataPath']
testing_dataPath = paths['testing_dataPath']
prob_Path = paths['prob_Path']
dataset_split_image1 = paths['dataset_split_image1']
dataset_split_image2 = paths['dataset_split_image2']
cm_image = paths['cm_image']
rpf1_image = paths['rpf1_image']
missclassification_result_image = paths['missclassification_result_image']
ccConfidence_image = paths['ccConfidence_image']
icConfidence_image = paths['icConfidence_image']
costMatrixPath = paths['costMatrixPath']


# load the cost matrix data
df = pd.read_csv(costMatrixPath)
CostMatrixData = df.to_dict('records')

# get the metrics and save them as images
total_train_images, total_test_images = utils_metrics.bar_plot_dataset_distribution(training_dataPath, testing_dataPath,dataset_split_image1)
utils_metrics.pie_plot_dataset_distribution(training_dataPath, testing_dataPath, dataset_split_image2)
utils_metrics.plot_confusion_matrix(prob_Path, cm_image)
accuracy = utils_metrics.plot_rpf1(prob_Path, rpf1_image)
utils_metrics.plot_cc_confidence(prob_Path, ccConfidence_image)
utils_metrics.plot_ic_confidence(prob_Path, icConfidence_image)
total_risk_score = utils_metrics.plot_misclassification_cost_matrix(prob_Path, missclassification_result_image, costMatrixPath)

try:
    dataset_split_image1 = InlineImage(template, dataset_split_image1, width=Mm(150))
    dataset_split_image2 = InlineImage(template, dataset_split_image2, width=Mm(150))
    cm_image = InlineImage(template, cm_image, width=Mm(150))
    rpf1_image = InlineImage(template, rpf1_image, width=Mm(120))
    missclassification_result_image = InlineImage(template, missclassification_result_image, width=Mm(150))
    ccConfidence = InlineImage(template, ccConfidence_image, width=Mm(140))
    icConfidence = InlineImage(template, icConfidence_image, width=Mm(140))

except Exception as e:
    logging.error(f'Error loading image: {e}')
    raise

# Define the context dictionary with the values to replace the placeholders
context = {
    'Model': 'InceptionV1', # probability table should be save with model name included so we can parse to include here
    'TrainingDate': date, 
    'DatasetID': '220525', # parse from the folder name
    'accuracy': round(accuracy*100, 2), 
    'riskscore': total_risk_score,
    'CostMatrix': CostMatrixData,
    'DataPathTrain': training_dataPath,
    'DataPathTest': testing_dataPath,
    'NumTrainImages' : total_train_images, 
    'NumTestImages' : total_test_images,
    'DatasetSplit1': dataset_split_image1,
    'DatasetSplit2': dataset_split_image2,
    'ConfusionMatrix': cm_image,
    'rpf1': rpf1_image,
    'MisclassificationCostPerClass': missclassification_result_image,
    'CCconfidence':ccConfidence,
    'ICconfidence':icConfidence,
    'docName': 'Results of 220525_InceptionV1',
    }

# Render the template with the context dictionary
try:
    template.render(context)
except Exception as e:
    logging.error(f'Error rendering template: {e}')
    raise

outputPath = 'reportOutput/output.docx'
# Save the rendered document to a new file
try:
    template.save(outputPath)
    # close the word document
    # template.close()
except Exception as e:
    logging.error(f'Error saving document: {e}')
    raise

# convert the pdf to a word document
pdfpath = outputPath[:-4]+'pdf'
try:
    convert(outputPath, pdfpath)
except Exception as e:
    logging.error(f'Error converting to pdf: {e}')
    raise











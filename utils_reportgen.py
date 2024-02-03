from docxtpl import DocxTemplate, InlineImage
from docxtpl import InlineImage
from docx.shared import Mm
import logging
import datetime
import pandas as pd
import utils_metrics
import yaml
import os


def generate_report(training_dataPath, testing_dataPath, prob_Path, output_path):
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # get the paths from the app
    training_dataPath = training_dataPath
    testing_dataPath = testing_dataPath
    prob_Path = prob_Path
    output_path = output_path

    # get the last part of the prob_Path to get the datasetID and model name
    last_part = os.path.basename(prob_Path)

    # get model base name
    model_base_name = last_part.split('-')[0]
    
    # get the model name
    # remove the last 4 characters (.csv)
    model_name = last_part[:-4]

    # get the dataset ID
    datasetID_1 = last_part.split('-')[1]
    datasetID_2 = last_part.split('-')[2]
    datasetID = datasetID_1 + '-' + datasetID_2

    # get the run number
    run_number = last_part.split('-')[4].split('.')[0]

    # define the output file name
    output_fileName = '/'+ model_base_name + '-' + datasetID + '-run-' + run_number +'.docx'
    
    # load the paths from the YAML file
    with open('def_file_paths.yml', 'r') as f:
        paths = yaml.safe_load(f)

    # access the paths using the dictionary keys
    dataset_split_image1 = paths['dataset_split_image1']
    dataset_split_image2 = paths['dataset_split_image2']
    cm_image = paths['cm_image']
    rpf1_image = paths['rpf1_image']
    pfmea_image = paths['pfmea_image']
    missclassification_result_image = paths['missclassification_result_image']
    ccConfidence_image = paths['ccConfidence_image']
    icConfidence_image = paths['icConfidence_image']
    costMatrixDirPath = paths['costMatrixDirPath']
    template_path = paths['template_path']

    # Load the Word document template
    template = DocxTemplate(template_path)

    # get date time
    now = datetime.datetime.now()

    # convert to uk format
    date = now.strftime("%d/%m/%Y")

    # get the list of files from the costMatrixDirPath
    files = os.listdir(costMatrixDirPath)
    
    # get the file that matches the datasetID_2
    matching_file = [s for s in files if datasetID_2 in s]
    
    # check if there are any files that match the datasetID_2
    if not matching_file:
        raise Exception(f"No files found in {costMatrixDirPath} that match {datasetID_2}")

    costMatrixPath = costMatrixDirPath + matching_file[0]

    # load the cost matrix data
    # df = pd.read_csv(costMatrixPath)
    # cost_matrix_data = df.to_dict('records')

    # get the metrics and save them as images
    total_train_images, total_test_images = utils_metrics.bar_plot_dataset_distribution(training_dataPath, testing_dataPath,dataset_split_image1)
    utils_metrics.pie_plot_dataset_distribution(training_dataPath, testing_dataPath, dataset_split_image2)
    utils_metrics.plot_confusion_matrix(prob_Path, cm_image)
    accuracy = utils_metrics.plot_rpf1(prob_Path, rpf1_image)
    utils_metrics.plot_cc_confidence(prob_Path, ccConfidence_image)
    utils_metrics.plot_ic_confidence(prob_Path, icConfidence_image)
    total_risk_score = utils_metrics.plot_misclassification_cost_matrix(prob_Path, missclassification_result_image, costMatrixPath, pfmea_image)

    try:
        dataset_split_image1 = InlineImage(template, dataset_split_image1, width=Mm(150))
        dataset_split_image2 = InlineImage(template, dataset_split_image2, width=Mm(150))
        cm_image = InlineImage(template, cm_image, width=Mm(150))
        rpf1_image = InlineImage(template, rpf1_image, width=Mm(120))
        pfmea_image = InlineImage(template, pfmea_image, width=Mm(150))
        missclassification_result_image = InlineImage(template, missclassification_result_image, width=Mm(150))
        ccConfidence = InlineImage(template, ccConfidence_image, width=Mm(140))
        icConfidence = InlineImage(template, icConfidence_image, width=Mm(140))

    except Exception as e:
        logging.error(f'Error loading image: {e}')
        raise

    # Define the context dictionary with the values to replace the placeholders
    context = {
        'Model': model_name, # probability table should be save with model name included so we can parse to include here
        'TrainingDate': date, 
        'DatasetID': datasetID, # parse from the folder name
        'accuracy': round(accuracy*100, 2), 
        'riskscore': total_risk_score,
        'CostMatrix': pfmea_image,
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
        'docName': 'Results of: '+ datasetID+'-'+ model_name,
    }

    # Render the template with the context dictionary
    try:
        template.render(context)
    except Exception as e:
        logging.error(f'Error rendering template: {e}')
        raise
    
    # Save the rendered document to a new file
    try:
        template.save(output_path+output_fileName)
    except Exception as e:
        logging.error(f'Error saving document: {e}')
        raise
    return output_path+output_fileName

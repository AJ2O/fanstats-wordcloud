# argparse
import argparse
parser = argparse.ArgumentParser(
    description='Loads social media posts on sports teams and uploads them to' 
    + ' AWS for further processing via Glue & Athena.')
parser.add_argument(
    '-r',
    '--twitter_results_per_page', 
    metavar='N', 
    type=int, 
    default=100,
    help='The number of Twitter results to retrieve per page request')
parser.add_argument(
    '-m',
    '--twitter_max_results', 
    metavar='N',
    type=int, 
    default=3000,
    help='The maximum number of Twitter results to retrieve')
args = parser.parse_args()

# constants / environment variables
ATHENA_CATALOG = ''
ATHENA_DB = ''
ATHENA_QUERY_RESULTS = ''
ATHENA_WORKGROUP = ''
LOCAL_OUTPUT_PATH = 'complete_wordclouds'

# import libraries
import os
import yaml

# math libraries
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_gradient_magnitude

# visualization libraries
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# time libraries
import datetime
from time import gmtime, sleep, strftime

# AWS clients
import boto3
athena_client = boto3.client('athena')

# file helpers
def load_environment_variables():
    if not os.environ.get('ENVIRONMENT_SETUP'):
        # TODO: download config file
        with open('config.yaml') as file:
            config = yaml.safe_load(file)
            os.environ.update(config)

def load_query_configuration(query_file):
    with open(query_file) as file:
        documents = list(yaml.safe_load_all(file))
        version = documents[0]
        universal_settings = documents[1]
        query_documents = documents[2:]
        return version, universal_settings, query_documents

# Athena helpers
def run_athena_query(query_string):
    '''
    Executes an Athena query using the given string, and returns a pandas
    dataframe containing the results.
    '''
    query_execution_id = athena_client.start_query_execution(
        QueryString = query_string,
        QueryExecutionContext = {
            'Catalog': ATHENA_CATALOG,
            'Database': ATHENA_DB
        },
        ResultConfiguration = {
            'OutputLocation': "s3://" + ATHENA_QUERY_RESULTS
        }
    )['QueryExecutionId']
    print("Executing Query with ID: " + query_execution_id + "...")
    
    # wait until the query returns results
    time_interval = 5
    time_iterations = 60
    for i in range(time_iterations):
        query_execution = athena_client.get_query_execution(
            QueryExecutionId = query_execution_id
        )
        # check query status
        query_status = query_execution['QueryExecution']['Status']['State']
        if query_status == 'SUCCEEDED':
            break
        elif query_status == 'CANCELLED':
            raise("The Athena query was cancelled: " 
                + query_execution['QueryExecution']['Status']['StateChangeReason'])

        if i == time_iterations:
            raise("The Athena query timed out after " 
                + str(time_interval * time_iterations) 
                +  " seconds")

        sleep(time_interval)

    # accumulate results
    query_results = []
    res_athena = athena_client.get_query_results(
        QueryExecutionId=query_execution_id
    )
    while len(res_athena['ResultSet']['Rows']) > 0:
        query_results.extend(res_athena['ResultSet']['Rows'])
        if 'NextToken' in res_athena:
            res_athena = athena_client.get_query_results(
                QueryExecutionId = query_execution_id,
                NextToken = res_athena['NextToken']
            )
        else:
            break
    
    # convert to pandas dataframe
    clean_result = [[column.get('VarCharValue') for column in row['Data']] 
        for row in query_results]
    df = pd.DataFrame(clean_result[1:], columns=clean_result[0])
    return df

# WordCloud main
def generate_wordcloud(text_list, mask_image_path=None, stopwords={}):
    # configure image masking
    mask = None
    if mask_image_path:
        image_mask = np.array(Image.open(mask_image_path))
        transformed_mask = image_mask.copy()
        transformed_mask[transformed_mask.sum(axis=2) == 0] = 255

        # enforce boundaries between colors so they get less washed out
        edges = np.mean([
            gaussian_gradient_magnitude(image_mask[:, :, i] / 255., 2)
            for i in range(3)], 
            axis=0)
        transformed_mask[edges > .08] = 255

    # generate a word cloud image
    aggregated_text = ". ".join(text_list)
    wordcloud = WordCloud(
        max_words=3000,
        mask=transformed_mask,
        random_state=5,
        stopwords=stopwords).generate(aggregated_text)
    
    # recolor the wordcloud based on the original image's colors
    if mask_image_path:
        image_colors = ImageColorGenerator(image_mask)
        wordcloud.recolor(color_func=image_colors)
    '''
    # plot the generated image
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    
    # Original image
    plt.figure(figsize=(10, 10))
    plt.title("Original Image")
    plt.imshow(image_mask)

    # Edge Map
    plt.figure(figsize=(10, 10))
    plt.title("Edge map")
    plt.imshow(edges)

    # display image
    plt.axis("off")
    plt.show()
    plt.close()
    '''
    return wordcloud

# WordCloud helpers
def configure_mask(mask_settings):
    mask_path = ''
    if mask_settings['type'] == 'local':
        return mask_settings['location']
    else:
        raise('Masks from ' + mask_settings['type'] + ' are unsupported')

def generate_document_wordcloud(
    document,
    text_list=[],
    universal_query_settings={}
    ):
    # mask
    mask_image_path = None
    if 'mask' in document:
        mask_image_path = configure_mask(document['mask'])
    elif 'mask' in universal_query_settings:
        mask_image_path = configure_mask(universal_query_settings['mask'])

    # stop words
    stopwords = set(STOPWORDS)
    if 'ignore' in universal_query_settings:
        stopwords.update(universal_query_settings['ignore'])
    if 'ignore' in document:
        stopwords.update(document['ignore'])

    return generate_wordcloud(
        text_list,
        mask_image_path, 
        stopwords)

def export_wordcloud(wordcloud, output_file_path):
    # create output path
    if not os.path.exists('complete_wordclouds'):
        os.mkdir('complete_wordclouds')
    full_output_path = LOCAL_OUTPUT_PATH + '/' + output_file_path
    
    # write wordcloud locally
    wordcloud.to_file(full_output_path)

    # TODO: upload to S3

# main
def run_query_document_v1_0(
    document,
    universal_query_settings={}
    ):
    query_type = document['type']
    
    # League
    if query_type == 'League':
        league = document['name']

        # run Athena query
        query_string = '''
        SELECT * FROM posts_by_league WHERE league = '{}'
        '''.format(league)
        df = run_athena_query(query_string)

        if not df.empty:
            # generate WordCloud
            wordcloud = generate_document_wordcloud(
                document,
                text_list=df.text,
                universal_query_settings=universal_query_settings)

            # export results
            output_file = league + '_' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.png'
            export_wordcloud(wordcloud, output_file)

    # Team
    elif query_type == 'Team':
        team = document['name']

        # run Athena query
        query_string = '''
        SELECT * from posts_by_team WHERE team = '{}'
        '''.format(team)
        df = run_athena_query(query_string)

        if not df.empty:
            # generate WordCloud
            wordcloud = generate_document_wordcloud(
                document,
                text_list=df.text,
                universal_query_settings=universal_query_settings)

            # export results
            output_file = team + '_' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.png'
            export_wordcloud(wordcloud, output_file)

    else:
        raise NotImplementedError("Query type " 
            + query_type 
            + " is unsupported")

def run_query_document(
    document,
    universal_query_settings={},
    queryfile_version=1.0
    ):
    if queryfile_version == 1.0:
        run_query_document_v1_0(document, universal_query_settings)
    else:
        raise NotImplementedError("Queryfile version "
            + queryfile_version 
            + " is unsupported")

def main():
    global ATHENA_CATALOG
    global ATHENA_DB
    global ATHENA_QUERY_RESULTS

    # set environment variables
    load_environment_variables()
    ATHENA_CATALOG = os.environ.get('ATHENA_CATALOG')
    ATHENA_DB = os.environ.get('ATHENA_DB')
    ATHENA_QUERY_RESULTS = os.environ.get('ATHENA_QUERY_RESULTS')

    # load queryfile
    query_file = os.environ.get('QUERY_FILE')
    queryfile_version, universal_query_settings, query_documents \
        = load_query_configuration(query_file)
    print("Queryfile Version: " + str(queryfile_version['version']))
    print("Universal query settings: " + str(universal_query_settings))

    # query all documents
    for document in query_documents:
        run_query_document(
            document, 
            universal_query_settings=universal_query_settings, 
            queryfile_version=queryfile_version['version'])

if __name__ == "__main__":
    main()
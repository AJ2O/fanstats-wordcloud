# Import basic libraries
import os
import re
import yaml
import urllib.request
import shutil

# Import statistics and machine learning libraries
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.ndimage import gaussian_gradient_magnitude
# The following NLTK resource only needs to be downloaded once. If it is
# already installed, the following two lines can be commented out.
import nltk
nltk.download('wordnet')

# Import visualization libraries
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Import time-related libraries
import datetime
from time import gmtime, sleep, strftime

# Import AWS SDK and configure service clients
import boto3
athena_client = boto3.client('athena')

# Define constants and environment variables
ATHENA_CATALOG = ''
ATHENA_DB = ''
ATHENA_QUERY_RESULTS = ''
ATHENA_WORKGROUP = ''
LOCAL_OUTPUT_PATH = 'complete_wordclouds'


# Configuration helpers
def load_environment_variables():
    '''
    Loads variables into the running environment, either from the host 
    environment or a local file. 
    '''
    if not os.environ.get('ENVIRONMENT_SETUP'):
        # TODO: download config file
        with open('config.yaml') as file:
            config = yaml.safe_load(file)
            os.environ.update(config)

def load_query_configuration(queryfile):
    '''
    Returns the configurations for the given queryfile.
    '''
    with open(queryfile) as file:
        documents = list(yaml.safe_load_all(file))
        version = documents[0]
        universal_settings = documents[1]
        query_documents = documents[2:]
        return version, universal_settings, query_documents


# AWS helpers
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
    print("Query " + query_execution_id + " complete!")

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


# WordCloud helpers
def load_template_mask(mask_settings):
    '''
    Returns a file path to an image mask.
    '''
    mask_path = ''
    mask_type = mask_settings['type']
    
    # the image is a stock image included with the project
    if mask_type == 'local':
        return mask_settings['location']
    
    # the image can be downloaded over the Internet
    elif mask_type == 'http':
        url = mask_settings['location']
        save_path = 'img/' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.png'
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        return save_path

    # no other images supported
    else:
        raise('Masks from ' + mask_type + ' are unsupported')

def load_template_stopwords(stopwords_settings):
    '''
    Returns a set of stopwords based on the given settings.
    '''
    stopwords = set()
    for setting in stopwords_settings:
        setting_type = setting['type']
        if setting_type == 'local':
            words = [line.strip() for line in open(setting['location'])]
            stopwords.update(words)
        else:
            raise NotImplementedError("Stopwords of type " 
                + setting_type 
                + " are unsupported")
    return stopwords

def load_template_wordcloud_config(document, universal_wc_config):
    '''
    Returns the WordCloud configurations for a given document.
    '''
    wc_max_words = int(universal_wc_config['max_words'])
    wc_random_state = int(universal_wc_config['random_state'])

    if 'wordcloud' in document:
        wc_config = document['wordcloud']
        if 'max_words' in wc_config:
            wc_max_words = int(wc_config['max_words'])
        if 'random_state' in wc_config:
            wc_random_state = int(wc_config['random_state'])

    return wc_max_words, wc_random_state

def normalize_text(text_list, stopwords={}):
    '''
    Pre-process and normalize raw text for cleanliness.
    '''
    normalized_text_list = []
    for original_text in text_list:
        # remove punctuation
        text = re.sub('[^a-zA-Z0-9]', ' ', original_text)

        # convert to lowercase
        text = text.lower()

        # remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

        # lemmatize text
        text = text.split()
        wnl = WordNetLemmatizer()
        text = [wnl.lemmatize(word) for word in text if not word in stopwords]

        # add normalized text
        text = " ".join(text)
        normalized_text_list.append(text)
        
    return normalized_text_list


# WordCloud
def generate_wordcloud(
    text_list, 
    mask_image_path=None,
    max_words=3000,
    random_state=5):
    '''
    Generates a WordCloud for the given text, and using any specified
    custom settings.
    '''
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

    # generate a WordCloud
    aggregated_text = " ".join(text_list)
    wordcloud = WordCloud(
        max_words=max_words,
        mask=transformed_mask,
        random_state=random_state).generate(aggregated_text)
    
    # recolor the WordCloud based on the original image's colors
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

def generate_document_wordcloud(
    document,
    text_list=[],
    universal_query_settings={}
    ):
    '''
    Generates a WordCloud for a given document and accompanying text.
    '''
    # mask
    mask_image_path = None
    if 'mask' in document:
        mask_image_path = load_template_mask(document['mask'])
    elif 'mask' in universal_query_settings:
        mask_image_path = load_template_mask(universal_query_settings['mask'])

    # stop words
    stopwords = set(STOPWORDS)
    if 'stopwords' in universal_query_settings:
        stopwords.update(load_template_stopwords(
            universal_query_settings['stopwords']
        ))
    if 'stopwords' in document:
        stopwords.update(load_template_stopwords(
            document['stopwords']
        ))

    # WordCloud configuration
    wc_max_words, wc_random_state = load_template_wordcloud_config(
        document=document,
        universal_wc_config=universal_query_settings['wordcloud']
    )

    # format text
    formatted_text = normalize_text(text_list, stopwords)

    return generate_wordcloud(
        formatted_text,
        mask_image_path=mask_image_path,
        max_words=wc_max_words,
        random_state=wc_random_state)

def export_wordcloud(wordcloud, output_file_path):
    '''
    Exports a wordcloud to the given file path.
    '''
    # create output path
    if not os.path.exists('complete_wordclouds'):
        os.mkdir('complete_wordclouds')
    full_output_path = LOCAL_OUTPUT_PATH + '/' + output_file_path
    
    # write wordcloud locally
    wordcloud.to_file(full_output_path)

    # TODO: upload to S3
    print("Exported wordcloud to " + output_file_path)


# Main
def run_query_document_v1_0(
    document,
    universal_query_settings={},
    source_df=None
    ):
    query_type = document['type']
    
    # League
    if query_type == 'League':
        league = document['name']

        if source_df is None:
            # run Athena query
            query_string = '''
            SELECT * FROM posts_by_league WHERE league = '{}'
            '''.format(league)
            df = run_athena_query(query_string)
        else:
            df = source_df.loc[source_df.league == league]

        if not df.empty:
            # generate WordCloud
            print("Generating WordCloud...")
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

        if source_df is None:
            # run Athena query
            query_string = '''
            SELECT * from posts_by_team WHERE team = '{}'
            '''.format(team)
            df = run_athena_query(query_string)
        else:
            df = source_df.loc[source_df.team == team]

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
    queryfile_version=1.0,
    source_df=None
    ):
    if queryfile_version == 1.0:
        print("Loading document " + document['name'] + "...")
        run_query_document_v1_0(document, universal_query_settings, source_df)
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

    # if the mode is 'ALL', run the query once, generate the dataframe,
    # and use it as the basis of truth for analyzing each document
    general_settings = universal_query_settings['general']
    source_df = None
    if general_settings['mode'] == 'ALL':
        source_df = run_athena_query(general_settings['query'])

    # query all documents
    for document in query_documents:   
        run_query_document(
            document, 
            universal_query_settings=universal_query_settings, 
            queryfile_version=queryfile_version['version'],
            source_df=source_df)

if __name__ == "__main__":
    main()
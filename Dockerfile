FROM python:3.9

WORKDIR /usr/src/app

COPY ./src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# provide environment variables securely
ENV QUERY_FILE=queryfiles/nba.yaml
ENV ATHENA_CATALOG=AwsDataCatalog
ENV ATHENA_DB=fsdb
ENV ATHENA_QUERY_RESULTS=s3_path
ENV ENVIRONMENT_SETUP=True

COPY ./src .
CMD [ "python", "./main.py" ]

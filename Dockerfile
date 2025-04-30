# FROM astrocrpublic.azurecr.io/runtime:3.0-1

FROM quay.io/astronomer/astro-runtime:12.6.0

RUN pip install apache-airflow-providers-google
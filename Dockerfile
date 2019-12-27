FROM python:3.7-slim
COPY . /app
ENV PYTHONPATH "/app:${PYTHONPATH}"
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000

# Define environment variable
ENV MODEL_NAME MyModel
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

RUN sed -i '4iimport Network' /usr/local/bin/seldon-core-microservice
RUN sed -i '5imodel_name = Network.__all__[0]' /usr/local/bin/seldon-core-microservice
RUN sed -i '6ilocals()[model_name] = getattr(Network, model_name)' /usr/local/bin/seldon-core-microservice

CMD python /usr/local/bin/seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE

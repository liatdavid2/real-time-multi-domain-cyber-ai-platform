FROM apache/spark:3.5.0

WORKDIR /app

# switch to root to create ivy directory
USER root
RUN mkdir -p /home/spark/.ivy2/cache
RUN chown -R spark:spark /home/spark/.ivy2

# switch back to spark user
USER spark

COPY spark_stream.py .
COPY schema.py .

CMD ["/opt/spark/bin/spark-submit", "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0", "spark_stream.py"]
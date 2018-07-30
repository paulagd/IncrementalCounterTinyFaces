# our base image
FROM avnergoncalves/ubuntu-python3.5

# specify the port number the container should expose
EXPOSE 5000

# run the application
#CMD ["python", "./app.py"]
RUN git clone https://github.com/paulagd/IncrementalCounterTinyFaces.git

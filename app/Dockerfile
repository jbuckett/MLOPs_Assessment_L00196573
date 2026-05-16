# lightweight version of python
FROM python:3.12-slim
# make a folder inside the container called /app and work from there
WORKDIR /app
# copy just the requirements file first 
COPY requirements.txt .
# install all the libraries I need
RUN pip install --no-cache-dir -r requirements.txt
# copy all my python files, the life expectancy csv, and the templates into the container
COPY . .
# The training step (RUN python train.py) has been removed for CI/CD deployment.
# The container will now serve the model.pkl file that is already saved in the repository.
# open up port 5000 so we can access the web page
EXPOSE 5000
# when the container starts up, run the flask app
CMD ["python", "app.py"]


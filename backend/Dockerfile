# start from official Python 3.12 image
FROM python:3.12-slim

# set working directory inside container
WORKDIR /app

# copy requirements first — Docker caches this layer
COPY requirements.txt .

# install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the backend code
COPY . .

# expose port 8000
EXPOSE 8000

# command to run when container starts
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
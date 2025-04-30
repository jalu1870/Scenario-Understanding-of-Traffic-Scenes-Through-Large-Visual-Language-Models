# Use an official Python runtime as a base image
FROM pytorch/pytorch:latest
# FROM python:3.10.12
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

RUN pip install --upgrade pip



RUN apt -y update
RUN apt -y install gcc

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application with arguments (override with docker run)
# CMD ["pip list"]
# CMD ["gcc", "--version"]
CMD ["python", "main.py", "--input_path", "tmp", "--task","predict", "--model", "llava-1.5", "--output_path", "data/out"]

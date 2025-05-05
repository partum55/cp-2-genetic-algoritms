# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE %PORT

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]
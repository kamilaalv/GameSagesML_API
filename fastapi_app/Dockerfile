# Use an official Python runtime as a parent image
FROM python:3.12-slim as build

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and the pre-saved model
COPY . .

# Final stage
FROM public.ecr.aws/lambda/python:3.12

# Set the working directory
WORKDIR /var/task

# Copy dependencies and the application from the build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /var/task
COPY --from=build /app /var/task

# Ensure the model is correctly set and cache directories are set to /tmp
ENV TRANSFORMERS_CACHE=/tmp
ENV HF_HOME=/tmp

# Define the CMD
CMD ["app.main.handler"]

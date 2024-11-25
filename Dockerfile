# Base Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the app's port
EXPOSE 5000

# Run the app
CMD ["python", "run.py"]
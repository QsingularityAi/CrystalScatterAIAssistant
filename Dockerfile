FROM python:3.9
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
# Copy the requirements file into the container
COPY app.py requirements.txt chainlit.md ./
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY . .

# Run app.py when the container launches
CMD ["chainlit", "run", "app.py","--port", "7860"]

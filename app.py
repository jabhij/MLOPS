from flask import flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


application = flask(__name__)   # Check point-- Flask\

app = application

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')


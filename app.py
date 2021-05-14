from flask import Flask, render_template, request, redirect
import os

from nltk.util import pr

import summarizer
import dates

app = Flask(__name__)


class Patient:
    def __init__(self, x, y):
        self.name = x
        self.date = y


files = []
for patient in dates.datesDict:
    files.append(Patient(patient, dates.datesDict[patient]))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/reports')
def reports():
    return render_template('reports.html', files=files)


@app.route('/view/<string:id>')
def view(id):
    report = summarizer.filecontent(id)
    print("------------------------------\n")
    print(report)
    return render_template('view.html', patient=id, report=report)


@app.route('/summarization')
def summarization():
    return render_template('summary.html', files=files)


@app.route('/summarize/<string:id>')
def summarize(id):
    summary = summarizer.summarisefile(id)
    return render_template('singleSummary.html', patient=id, summary=summary)


@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')


if __name__ == "__main__":
    app.run(debug=True)

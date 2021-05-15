import datetime
from flask import Flask, render_template, request, redirect
import os

from nltk.util import pr

import summarizer
import dates
import searchString

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


@app.route('/summarization', methods=['POST', 'GET'])
def summarization():
    if request.method == 'POST':
        print('in post')
        query = request.form['search']
        start = request.form['start']
        end = request.form['end']
        submit = None
        reset = None
        flag = False
        try:
            submit = request.form['Submit']
            flag = True
        except:
            print('submit unrecognized')
        try:
            reset = request.form['Reset']
            flag = False
        except:
            print('reset unrecognized')
        result = {}
        print("-----------", start, end, type(start))
        if flag:
            if query == '':
                result = dates.datesDict
            else:
                result = searchString.search(query)

            if start == '' and end == '':
                pass
            elif start == '' and end != '':
                values = [int(x) for x in end.split('-')]
                result = dates.betweenStartandEnd(result, datetime.datetime(
                    1990, 1, 1), datetime.datetime(values[0], values[1], values[2]))
            elif start != '' and end == '':
                values = [int(x) for x in start.split('-')]
                result = dates.betweenStartandEnd(result, datetime.datetime(
                    values[0], values[1], values[2]), datetime.datetime(2025, 1, 1))
            else:
                startValues = [int(x) for x in start.split('-')]
                endValues = [int(x) for x in end.split('-')]
                result = dates.betweenStartandEnd(result, datetime.datetime(
                    startValues[0], startValues[1], startValues[2]), datetime.datetime(endValues[0], endValues[1], endValues[2]),)
            queryResults = []
            for patient in result:
                queryResults.append(Patient(patient, dates.datesDict[patient]))
            return render_template('summary.html', files=queryResults, query=query, start=start, end=end)
        else:
            return render_template('summary.html', files=files, query='', start='', end='')
    else:
        return render_template('summary.html', files=files, query='', start='', end='')


@app.route('/summarize/<string:id>')
def summarize(id):
    summary = summarizer.summarisefile(id)
    return render_template('singleSummary.html', patient=id, summary=summary)


@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')


if __name__ == "__main__":
    app.run(debug=True)

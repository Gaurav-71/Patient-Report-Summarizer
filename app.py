from flask import Flask, render_template, request, redirect

app = Flask(__name__)


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
    return render_template('reports.html')


@app.route('/summarization')
def summarization():
    return render_template('summary.html')


@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')


if __name__ == "__main__":
    app.run(debug=True)

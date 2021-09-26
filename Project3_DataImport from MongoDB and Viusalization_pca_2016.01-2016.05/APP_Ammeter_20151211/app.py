from flask import Flask, g, jsonify, url_for
from flask import render_template
from flask.ext.pymongo import PyMongo

app = Flask(__name__)

app.config['MONGO_HOST'] = 'localhost'
app.config['MONGO_PORT'] = 27017
app.config['MONGO_USERNAME'] = 'localhost-am'
app.config['MONGO_PASSWORD'] = 'BGnd03kntrHL'
app.config['MONGO_DBNAME'] = 'am'
mongo = PyMongo(app)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stat')
def stat():
    url_for('static', filename='style.css')
    url_for('static', filename='d3.v3.min.js')
    url_for('static', filename='stat.js')
    return render_template('stat.html')

@app.route('/data/stat')
def fetchStat():
    kwh = mongo.db.Merged_Stat.find_one()
    return jsonify(kwh)

if __name__ == '__main__':
    app.run(debug=True)

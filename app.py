from flask import flask
import os

app=Flas(__name__)

@app.route('/')
def run_experiment():
    return "MLOPS Experiment service is running"

if __name__=="__main__":
    port=int(os.environ.get('PORT',8080))
    app.run(host='0.0.0.0',port=port)
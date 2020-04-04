import time
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/SMILES/<smiles>', methods=['GET'])
def get_smiles(smiles):
    return {'smiles': smiles, 'KNN': 'J', 'ANN': 'N', 'SVM': 'M', 'Actual': 'J'}

@app.route('/LINCS/<genes>', methods=['GET'])
def get_lincs(genes):
    return {'genes': genes, 'KNN': 'a', 'ANN': 'b', 'SVM': 'c', 'Actual': 'd'}

if __name__ == '__main__':
    app.run()

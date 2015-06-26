import os
from flask import Flask, render_template, request
from pricemyrental import PriceMyRental, load_data_and_models, run_pmr

# initialization
app = Flask(__name__)

#Load models necessary data
app.rfr, app.search_df, app.nhood_medians, app.vectorizer, app.nmf, app.kd = load_data_and_models()

# controllers
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/receive", methods=['POST'])
def receive():
    f = request.form

    beds         = float(f['beds'])
    baths        = float(f['baths'])
    address      = str(f['address']) + ' ' + str(f['zip'])
    neighborhood = str(f['neigh'])
    parking      = float(f['parking'])
    description  = str(f['desc'])
    price        = int(f['price'])

    predict_statement, compare_statement = run_pmr(beds, baths, address, neighborhood, parking, 
                                                   description, price, app.rfr, 
                                                   app.search_df, app.nhood_medians, 
                                                   app.vectorizer, app.nmf, app.kd)

    # Your code here to process data

    return render_template('result.html', prediction=predict_statement, 
                            comparison=compare_statement)



# launch
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
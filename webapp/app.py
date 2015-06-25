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

    beds         = int(f['beds'])
    baths        = int(f['baths'])
    address      = str(f['address'])
    neighborhood = str(f['neigh'])
    parking      = int(f['parking'])
    washer_dryer = int(f['washer_dryer'])
    description  = str(f['description'])
    price        = int(f['price'])

    predict_statement, compare_statement = run_pmr(beds, baths, address, neighborhood, parking, 
                                                   washer_dryer, description, price,
                                                   app.rfr, app.search_df, app.nhood_medians, 
                                                   app.vectorizer, app.nmf, app.kd)

    print predict_statement
    print compare_statement
    # Your code here to process data

    return render_template('index.html')

# launch
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
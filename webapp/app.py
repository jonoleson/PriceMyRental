import os
from flask import Flask, render_template, request

# initialization
app = Flask(__name__)

# controllers
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/receive", methods=['POST'])
def receive():
    f            = request.form
    n_beds       = f['# Bedrooms:']
    n_baths      = f['# Bathrooms:']
    address      = f['Address:']
    neighborhood = f['Neighborhood:']
    parking      = f['Parking:']
    washer_dryer = f['Washer-dryer in unit:']
    price        = f['Current/Intended listed price (for comparison):']
    description  = f['Description of unit (e.g, craigslist description):']
    print n_beds
    print n_baths
    print address
    print neighborhood
    print parking
    print washer_dryer
    print price
    print description
    # Your code here to process data

    return render_template('index.html')

# launch
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
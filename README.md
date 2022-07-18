# Python implementation of solution to the multiple traveling salesperson problem using Google's Operations Research tools library (OR-Tools)

- **Functionality**: given a list of physical addresses in CSV format, this code finds an approximate solution to the multiple traveling salesperson problem (mTSP) with a user-specified number of salespeople using. The approximate solution is obtained using the vehicle routing  methods in Google's OR-Tools library. The output of the code is a CSV file containing the optimal routes for each salesperson. It uses the Google Maps API to compute the distance matrix using real-time traffic information. 

- [Google OR-Tools Vehicle Routing](https://developers.google.com/optimization/routing/vrp)


## Usage instructions

- If you have never registered for an API key on the Google Maps Platform, follow [this](https://developers.google.com/maps/gmp-get-started) getting started guide. 
- [Create](https://developers.google.com/maps/documentation/distance-matrix/get-api-key?hl=en_US) an API key for the Distance Matrix API and save it somewhere.
- [Create](https://www.twilio.com/blog/2017/01/how-to-set-environment-variables.html) an environment variable for this API key. In the code, the environment variable name for this key is `GMAPS_KEY`.

- Save a CSV file containing your addresses in a single column with header 'Address' in the same directory as `main.py` and `utils.py`. In the `main` method in `main.py` change line 18 to  `addrs_file = 'your_csv_file_name.csv'`. 

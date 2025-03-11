from flask import Flask,render_template,jsonify
import csv
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ML','oop_t')))
from flightoop2 import find_optimal_route

app=Flask(__name__)

def get_route_data(optimal_route_id):
    with open(f"routes/{optimal_route_id}.csv",'r') as file:
        reader = csv.DictReader(file)
        route_data = [row for row in reader]
    return route_data

    

@app.route('/get-flight-path')
def get():
    # optimal_route_id = "AI2886_38a1bfd2"
    try:
        shortest_route = find_optimal_route()
        if not shortest_route or "No optimal route found" in shortest_route:
            return jsonify({"error": "No optimal route found"}), 404 
        
        optimal_route_id = shortest_route.split("\\")[-1].split(".")[0]

        route_data = get_route_data(optimal_route_id)
        if not route_data:
            return jsonify({"error": "Flight data not found"}),404
        
        return jsonify(route_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



    
@app.route('/')
def index():
    return render_template('h.html')


if __name__ == '__main__':
    app.run(debug=True)

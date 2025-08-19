from __future__ import print_function
import pandas as pd
import numpy as np
import glob
import os
import folium
from scipy.ndimage import gaussian_filter
from geopy.distance import geodesic
import time
import weatherapi
from pprint import pprint
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from joblib import dump, load
from collections import Counter
import requests
import json
from datetime import datetime
import cProfile
import pstats
import io



class FlightWeatherProcessor:
    def __init__(self, input_dir, output_dir, api_instance,key):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.api_instance = api_instance
        self.apikey=key
        self.common_columns = [
            'temp_c', 'humidity', 'wind_kph', 'wind_degree',
            'pressure_mb', 'cloud', 'feelslike_c', 'dewpoint_c',
            'vis_km', 'gust_kph', 'uv'
        ]
        self.MAX_RETRIES = 3  # Maximum number of retry attempts
        self.RETRY_DELAY = 10
        self.current_data=[]# Delay in seconds between retries

    def create_output_directory(self):
        """Create the output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def process_file(self, num, file_path):
        """Process a single file to fetch and add weather data."""
        print(f"Processing: {file_path}")
        df = pd.read_csv(file_path)

        # Add new weather columns if they don't exist
        for col in self.common_columns:
            if col not in df.columns:
                df[col] = None
        df["route_id"]=num
        for idx, row in df.iterrows():
            retries = 0
            while retries < self.MAX_RETRIES:
                try:
                    q = f"{row['lat']},{row['lon']}"
                    dt = row['timestamp']
                    t = time.gmtime(dt)
                    hour = t.tm_hour

                    # Get historical weather data
                    api_response = self.api_instance.history_weather(
                        q,
                        dt="",
                        unixdt=dt,
                        hour=hour
                    )

                    forecastday = api_response['forecast']['forecastday'][0]
                    # Extract relevant data from response
                    if forecastday:
                        hour_data = forecastday['hour'][0]

                        if hour_data:
                            # Update dataframe with weather data
                            df.at[idx, 'temp_c'] = hour_data['temp_c']
                            df.at[idx, 'humidity'] = hour_data['humidity']
                            df.at[idx, 'wind_kph'] = hour_data['wind_kph']
                            df.at[idx, 'wind_sin'] = np.sin(np.radians(hour_data['wind_degree']))
                            df.at[idx, 'wind_cos'] = np.cos(np.radians(hour_data['wind_degree']))
                            df.at[idx, 'pressure_mb'] = hour_data['pressure_mb']
                            df.at[idx, 'cloud'] = hour_data['cloud']
                            df.at[idx, 'feelslike_c'] = hour_data['feelslike_c']
                            df.at[idx, 'dewpoint_c'] = hour_data['dewpoint_c']
                            df.at[idx, 'vis_km'] = hour_data['vis_km']
                            df.at[idx, 'gust_kph'] = hour_data['gust_kph']
                            df.at[idx, 'uv'] = hour_data['uv']

                    break  # Success, exit retry loop

                except Exception as e:
                    retries += 1
                    if retries < self.MAX_RETRIES:
                        print(f"Error processing row {idx}: {e}. Retrying ({retries}/{self.MAX_RETRIES})...")
                        time.sleep(self.RETRY_DELAY)
                    else:
                        print(f"Failed processing row {idx} after {self.MAX_RETRIES} attempts: {e}")
                        # Optionally log the error or mark the row as failed
                        df.at[idx, 'processing_error'] = str(e)
                        continue

        # Save to new directory
        output_path = os.path.join(self.output_dir, os.path.basename(file_path))
        df.to_csv(output_path, index=False)
        print(f"Saved processed file to: {output_path}")

    def process_all_files(self,skip_existing=False):
        """Process all files in the input directory."""
        self.create_output_directory()
        files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        processed_files = set(os.path.basename(f) for f in
                              glob.glob(os.path.join(self.output_dir, "*.csv")))

        for rnum, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            output_path = os.path.join(self.output_dir, file_name)

            # Skip if the file has already been processed
            if file_name in processed_files and skip_existing:
                print(f"Skipping already processed file: {file_name}")
                continue
            self.process_file(rnum, file_path)


    def chunk_locations(self, locations, chunk_size=50):
        """Splits the locations list into smaller chunks of size 'chunk_size'."""
        return [locations[i:i + chunk_size] for i in range(0, len(locations), chunk_size)]

    def get_current_weather(self,input):
        standard_path = pd.read_csv(input)

        locations = []
        for idx, row in standard_path.iterrows():
            q = f"{row['latitude']},{row['longitude']}"
            custom_id = f"loc-{idx}"  # You can change this to use an existing column if desired
            locations.append({
                "q": q,
                "custom_id": custom_id
            })
        print(locations)
        location_chunks = self.chunk_locations(locations, chunk_size=50)
        bulk_response = []
        for i, chunk in enumerate(location_chunks, start=1):
            print(f"\n--- Processing chunk {i}/{len(location_chunks)} ---")
            response = self.fetch_current_weather(locations=chunk)

            if response:
                bulk_response.extend(response)



        if bulk_response:
            for num,item in enumerate(bulk_response):
                query = item.get("query", {})
                current = query.get("current", {})
                if current:

                    self.current_data.append([
                        standard_path.iloc[num]["latitude"],
                        standard_path.iloc[num]["longitude"],
                        current.get("temp_c"),
                        current.get("wind_kph"),
                        current.get("humidity"),
                        current.get("pressure_mb"),
                        current.get("cloud"),
                        np.sin(np.radians(current.get("wind_degree"))),
                        np.cos(np.radians(current.get("wind_degree"))),

                        current.get("dewpoint_c"),
                        # current.get("vis_km"),

                    ])


    def fetch_current_weather(self,locations):
        BULK_ENDPOINT = f"http://api.weatherapi.com/v1/current.json?key={self.apikey}&q=bulk"
        headers = {"Content-Type": "application/json"}
        payload = {"locations": locations}

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(BULK_ENDPOINT, headers=headers, data=json.dumps(payload))
                if response.status_code >= 400:
                    try:
                        error_info = response.json().get("error", {})
                        error_code = error_info.get("code")
                        error_message = error_info.get("message")
                        print(
                            f"❌ Attempt {attempt + 1}: Error {response.status_code} (code {error_code}) - {error_message}")
                    except Exception:
                        print(f"❌ Attempt {attempt + 1}: Error {response.status_code} - Unable to parse error details.")
                    response.raise_for_status()

                data = response.json()
                if "bulk" in data:
                    return data["bulk"]  # Return the bulk list from the response
                else:
                    print(f"⚠️ Attempt {attempt + 1}: 'bulk' field not found in response. Retrying...")
            except requests.exceptions.RequestException as e:
                print(f"❌ Attempt {attempt + 1}: Exception occurred - {e}")

            time.sleep(10)

        print("🚨 Max retries reached for bulk request. Skipping...")
        return None
    

    def filtered_files(self, start_datetime, end_datetime, output_filtered_dir, flight_number=None, skip_existing=False):
        #filter and save
        os.makedirs(output_filtered_dir, exist_ok=True)
        files = glob.glob(os.path.join(self.output_dir, "*.csv")) # self.output_dir is where original processed weather files are

        for file in files:
            file_name = os.path.basename(file)

            if flight_number and flight_number not in file_name:
                    print(f"Skipping {file_name}: flight {flight_number} not found in name")
                    continue

            output_path = os.path.join(output_filtered_dir, file_name)

            if skip_existing and os.path.exists(output_path):
                print(f"Skipping already filtered file: {file_name}")
                continue

            try:
                df = pd.read_csv(file)

                first_ts = pd.to_datetime(df['timestamp'].iloc[0], unit='s')
                if start_datetime <= first_ts <= end_datetime:
                    filtered_df = df  
                    filtered_df.to_csv(output_path, index=False)
                    print(f"Filtered and saved: {file_name} to {output_path}")
                else:
                    print(f"No data within range for {file_name}, skipping.")

            except Exception as e:
                print(f"Error filtering file {file_name}: {e}")




class FlightRouteProcessor:
    def __init__(self, folder_path, skip=False):
        self.folder_path = folder_path
        self.files = glob.glob(os.path.join(folder_path, "*.csv"))
        self.dfs = []
        self.skip = skip
        self.starting_points = []
        self.combined_df = None
        self.average_route = None

    def parse_position(self, position):
        """Convert position string to tuple of floats."""
        lat, lon = map(float, position.strip('"').split(','))
        return lat, lon

    def load_and_process_routes(self):
        if self.skip:
            return
        """Load all route files and compute cumulative distances."""
        for i, file in enumerate(self.files):
            df = pd.read_csv(file)
            df[['latitude', 'longitude']] = df['Position'].str.split(',', expand=True).astype(float)

            # Compute cumulative distance
            df['distance'] = [0] + [geodesic((df.iloc[j - 1]['latitude'], df.iloc[j - 1]['longitude']),
                                             (df.iloc[j]['latitude'], df.iloc[j]['longitude'])).meters
                                    for j in range(1, len(df))]
            df['distance'] = df['distance'].cumsum()

            df['route_id'] = i
            self.dfs.append(df)

            # Store starting points
            self.starting_points.append((df.iloc[0]['latitude'], df.iloc[0]['longitude']))

        self.combined_df = pd.concat(self.dfs)

    def compute_average_route(self):
        if self.skip:
            return
        """Compute the average route from all routes."""
        # Compute the common starting point as the average of all starting points
        common_start_lat = np.mean([p[0] for p in self.starting_points])
        common_start_lon = np.mean([p[1] for p in self.starting_points])

        # Interpolate latitude and longitude based on distance
        common_distances = np.linspace(0, self.combined_df['distance'].max(), num=100)
        average_latitudes = np.interp(common_distances, self.combined_df['distance'], self.combined_df['latitude'])
        average_longitudes = np.interp(common_distances, self.combined_df['distance'], self.combined_df['longitude'])

        # Create a DataFrame for the average route
        self.average_route = pd.DataFrame(
            {'distance': common_distances, 'latitude': average_latitudes, 'longitude': average_longitudes})

        # Apply Gaussian smoothing to reduce noise
        self.average_route['latitude'] = gaussian_filter(self.average_route['latitude'], sigma=3)
        self.average_route['longitude'] = gaussian_filter(self.average_route['longitude'], sigma=3)

        # Adjust the first point of the average route to the common starting point
        self.average_route.iloc[0, self.average_route.columns.get_loc('latitude')] = common_start_lat
        self.average_route.iloc[0, self.average_route.columns.get_loc('longitude')] = common_start_lon

    def visualize_routes(self):
        """Visualize all routes and the average route."""
        plt.figure(figsize=(10, 6))
        for route_id, group in self.combined_df.groupby('route_id'):
            plt.plot(group['longitude'], group['latitude'], label=f'Route {route_id}', alpha=0.5)
        plt.plot(self.average_route['longitude'], self.average_route['latitude'], label='Average Route', color='black',
                 linewidth=3)
        plt.legend()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Average Route')
        plt.show()

    def save_average_route(self, output_file):
        if self.skip:
            return
        """Save the average route to a CSV file."""

        #PLOT IN MAP AND VISUALISE
        m = folium.Map(location=[self.average_route['latitude'].mean(), self.average_route['longitude'].mean()], zoom_start=12)
        for i, row in self.average_route.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=1
            ).add_to(m)
        m.save('average_route_map.html')
        self.average_route.to_csv(output_file, index=False)



class FlightDataProcessor:
    def __init__(self, flights_dir, avg_route_path, output_dir):
        self.flights_dir = flights_dir
        self.avg_route_path = avg_route_path
        self.output_dir = output_dir

    def find_closest_point(self, avg_point, sample_points):
        """Find the closest point in `sample_points` to `avg_point`."""
        closest_point = min(sample_points, key=lambda x: geodesic(avg_point, x).meters)
        return closest_point

    def process_flight_data(self, flight_file):
        """Process a single flight file and save the result."""
        flight_path = os.path.join(self.flights_dir, flight_file)
        flight_data = pd.read_csv(flight_path)

        # Split 'Position' into 'lat' and 'lon' columns
        flight_data[['lat', 'lon']] = flight_data['Position'].str.split(",", expand=True)
        flight_data['lat'] = pd.to_numeric(flight_data['lat'])
        flight_data['lon'] = pd.to_numeric(flight_data['lon'])

        # Get flight points as a list of tuples
        flight_points = list(zip(flight_data['lat'], flight_data['lon']))

        # Load the average route
        avg_route = pd.read_csv(self.avg_route_path)
        route_points = list(zip(avg_route['latitude'], avg_route['longitude']))

        # Process each waypoint in the average route
        processed_data = []
        for wpId, row in enumerate(route_points):
            closest = self.find_closest_point(row, flight_points)
            index = flight_points.index(closest)

            # Extract relevant flight data
            processed_data.append({
                "timestamp": flight_data.loc[index]['Timestamp'],
                "lat": row[0],
                "lon": row[1],
                "altitude": flight_data.loc[index]['Altitude'],
                "speed": flight_data.loc[index]['Speed'],
                "direction": flight_data.loc[index]['Direction'],
                "waypoint_ID": wpId
            })

        # Save the processed data
        output_path = os.path.join(self.output_dir, flight_file)
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_csv(output_path, index=False)
        print(f"Saved processed file to: {output_path}")

    def process_all_flights(self,skip_existing=False):
        """Process all flight files in the input directory."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        flight_files = os.listdir(self.flights_dir)
        processed_files = set(os.listdir(self.output_dir))

        for flight_file in flight_files:
            # Skip if the file has already been processed
            if flight_file in processed_files and skip_existing:
                print(f"Skipping already processed flight file: {flight_file}")
                continue

            self.process_flight_data(flight_file)

class FlightClusterProcessor:
    def __init__(self, input_dir, output_dir, scalers_dir, pca_dir, models_dir, visualizations_dir):
        self.clusters = []
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.scalers_dir = scalers_dir
        self.pca_dir = pca_dir
        self.models_dir = models_dir
        self.visualizations_dir = visualizations_dir
        self.features = [
            "temp_c", "wind_kph", "humidity", "pressure_mb", "cloud",
            "wind_sin", "wind_cos", "dewpoint_c"
        ]

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
        os.makedirs(self.pca_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def find_optimal_clusters(self, X_train, max_k=11):
        """Find the optimal number of clusters using the elbow method and silhouette scores."""
        n_samples = len(X_train)    #no of csv files
        max_k = min(max_k, n_samples - 1)
        
        # If we have very few samples, return a small number of clusters
        if n_samples < 4:
            print(f"Too few samples ({n_samples}), using k=2")
            return 2
        if max_k < 2:
            print(f"max_k too small ({max_k}), using k=2")
            return 2
        
        scores = []
        silhouette_scores = {}
        valid_k_values = []
        
        # Test different numbers of clusters
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans_labels = kmeans.fit_predict(X_train)
                scores.append(kmeans.inertia_)
                silhouette_avg = silhouette_score(X_train, kmeans_labels)
                silhouette_scores[k] = silhouette_avg
                valid_k_values.append(k)
            except Exception as e:
                print(f"Error calculating clusters for k={k}: {e}")
                continue
        
        # If no valid clusters were found, return 2
        if not scores or not silhouette_scores or not valid_k_values:
            print("No valid clusters found, using k=2")
            return 2

        # Use KneeLocator to find the elbow point
        try:
            kl = KneeLocator(valid_k_values, scores, curve="convex", direction="decreasing")
            optimal_k = kl.elbow
            print(f"KneeLocator found elbow at k={optimal_k}")
        except Exception as e:
            print(f"KneeLocator error: {e}")
            optimal_k = None
        
        # If KneeLocator fails to find an elbow, fall back to the k with best silhouette score
        if optimal_k is None or optimal_k not in valid_k_values:
            if silhouette_scores:
                best_k = max(silhouette_scores.keys(), key=lambda k: silhouette_scores[k])
                print(f"KneeLocator failed, using k={best_k} based on silhouette score")
                return int(best_k)
            else:
                # Ultimate fallback - use 2 clusters
                print("Both KneeLocator and silhouette score failed, using k=2")
                return 2
        
        # Validate that optimal_k is a valid integer
        if optimal_k is None or not isinstance(optimal_k, (int, np.integer)) or optimal_k < 2:
            print(f"Invalid optimal_k value: {optimal_k}, using k=2")
            return 2
        
        final_k = int(optimal_k)
        print(f"Final k value: {final_k}")
        return final_k

    def form_clusters(self, wp, data):
        """Perform clustering for a specific waypoint."""
        n_data = data[self.features]

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(n_data.values)

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        # Save the scaler and PCA models
        dump(scaler, f'{self.scalers_dir}/scaler_{wp}.joblib')
        dump(pca, f'{self.pca_dir}/pca_{wp}.joblib')

        # Find the optimal number of clusters
        try:
            n_clusters = self.find_optimal_clusters(reduced_data)
            print(f"Waypoint {wp}: Using {n_clusters} clusters for {len(n_data)} samples")
        except Exception as e:
            print(f"Waypoint {wp}: Error finding optimal clusters: {e}. Using 2 clusters.")
            n_clusters = 2

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(reduced_data)

        # Save the KMeans model
        dump(kmeans, f'{self.models_dir}/waypoint_{wp}.joblib')

        # Add cluster labels to the data
        data["cluster"] = kmeans_labels

        data.to_csv(f'{self.output_dir}/waypoint_{wp}.csv', index=False)

    def process_all_waypoints(self):
        """Process all waypoints in the input directory."""
        self.create_directories()
        files = glob.glob(f"{self.input_dir}/*.csv")
        combined_df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

        for waypoint_id, group in combined_df.groupby("waypoint_ID"):
            self.form_clusters(waypoint_id, group)

    def predict_and_save_visualizations(self, current_data):
        """Predict clusters for current weather data and save visualizations."""
        for wp_num, current_features in enumerate(current_data):
            train_df = pd.read_csv(f'{self.output_dir}/waypoint_{wp_num}.csv')
            matched_train_point = self.match_location_with_train_data(train_df, current_features)  
            train_df = train_df[self.features + ["cluster"]]
            location_data=current_features
            current_features=current_features[2:]

            scaler = load(f'{self.scalers_dir}/scaler_{wp_num}.joblib')
            pca = load(f'{self.pca_dir}/pca_{wp_num}.joblib')
            kmeans = load(f'{self.models_dir}/waypoint_{wp_num}.joblib')

            # Process current weather data
            scaled_current = scaler.transform([current_features])
            reduced_current = pca.transform(scaled_current)
            cluster = kmeans.predict(reduced_current)[0]
            self.clusters.append(cluster)
            # Save visualization as an image

            GENERATE_PLOTS = False
            if GENERATE_PLOTS:
                self._save_cluster_visualization(wp_num, train_df, scaler, pca, kmeans, reduced_current, cluster)
                self.visualize_on_map(wp_num, location_data, matched_train_point)

    def match_location_with_train_data(self, train_df, current_features):
        """Match current weather data location with the closest training data point."""
        # Assuming current_features include [latitude, longitude, temp_c, wind_kph, ...]
        print(current_features)
        current_lat, current_lon = current_features[0], current_features[1]

        # Find the closest training data point
        train_latitudes = train_df['lat'].values
        train_longitudes = train_df['lon'].values

        # Calculate the geodesic distance (in meters) between the current point and each training point
        distances = [
            geodesic((current_lat, current_lon), (train_lat, train_lon)).meters
            for train_lat, train_lon in zip(train_latitudes, train_longitudes)
        ]

        # Find the index of the closest match
        min_distance_idx = np.argmin(distances)
        closest_train_point = train_df.iloc[min_distance_idx]

        return closest_train_point  # Return the matched training data point

    def visualize_on_map(self, wp_num, current_features, matched_train_point):
        """Visualize the current weather data and the matched training data on an interactive map."""
        # Create a map centered around the current weather point
        m = folium.Map(location=[current_features[0], current_features[1]], zoom_start=12)

        # Add the current weather point to the map
        folium.Circle(
            location=[current_features[0], current_features[1]],
            popup="Current Weather",
            radius=1000,
        ).add_to(m)

        # Add the matched training data point to the map
        folium.Marker(
            location=[matched_train_point['lat'], matched_train_point['lon']],
            popup="Matched Training Point",
            icon=folium.Icon(color='blue', icon='cloud')
        ).add_to(m)

        # Save the map as an HTML file
        m.save(f'{self.visualizations_dir}/waypoint_{wp_num}_map.html')
    def _save_cluster_visualization(self, wp_num, train_df, scaler, pca, kmeans, reduced_current, cluster):
        """Save cluster visualization as an image."""
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors

        for c in range(kmeans.n_clusters):
            cluster_data = train_df[train_df['cluster'] == c]
            reduced_train = pca.transform(scaler.transform(cluster_data.iloc[:, :-1].values))
            plt.scatter(reduced_train[:, 0], reduced_train[:, 1],
                        color=colors[c], label=f'Cluster {c}', alpha=0.6)

        # Plot current prediction
        plt.scatter(reduced_current[:, 0], reduced_current[:, 1],
                    color='red', marker='*', s=300,
                    label=f'Current (Cluster {cluster})', edgecolors='black')

        plt.title(f'Waypoint {wp_num} Cluster Visualization\n(PCA Reduced Space)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)

        # Save the plot as an image
        plt.savefig(f'{self.visualizations_dir}/waypoint_{wp_num}.png')
        plt.close()
# Main program
class RouteFinder:
    def __init__(self, cluster, input_dir, limit, routes_dir):
        self.limit=limit
        self.routes_dir=routes_dir
        self.input_dir=input_dir
        self.clusters=cluster
        self.possible_routes=[]
    def filter_waypoints(self,df, cluster_num):
        routes = []

        for num, i in df.iterrows():

            if i["cluster"] == cluster_num:
                routes.append(i["route_id"])
        self.possible_routes.extend(routes)

    def predict_optimal_route(self):
        for i in range(len(self.clusters)):
            df=pd.read_csv(f"{self.input_dir}/waypoint_{i}.csv")
            self.filter_waypoints(df, self.clusters[i])
        counter = Counter(self.possible_routes)
        common=counter.most_common(self.limit)
        return self.find_shortest_route(common)
    def find_shortest_route(self,common):
        routes=glob.glob(os.path.join(self.routes_dir, "*.csv"))
        #return shortest
        dict={}
        for route_id, count in common:
            print(f"Route ID: {route_id}, Count: {count} of {routes[int(route_id)]}")
            # Load the route file and calcute the one with the minumum time to destination
            df = pd.read_csv(routes[int(route_id)])
            #find the time to destination using gthe starting anf ending time
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
            df = df.sort_values('Timestamp')
            start_time = df.iloc[0]['Timestamp']
            end_time = df.iloc[-1]['Timestamp']
            time_to_destination = end_time - start_time
            print(f"Time to Destination: {time_to_destination}")
            dict[route_id]=time_to_destination
        min_route=min(dict, key=dict.get)
        print(f"Optimal Route: {min_route}, Time to Destination: {dict[min_route]}")
        return routes[int(min_route)]




def find_optimal_route(start_date_str, end_date_str, flight_number=None):
    # filtering out range
    # start_date_str = "2024-12-01"  
    # end_date_str = "2024-12-31"
    start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date_str, "%Y-%m-%d")


    key = 'c813413bbf4a41559ce50535251908'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    flights_dir = os.path.join(current_dir, "flights")

    # Print the resolved path
    print("Current Directory:", current_dir)
    print("Flights Directory:", flights_dir)

    filtered_base_dir = os.path.join(current_dir, "filtered")
    os.makedirs(filtered_base_dir, exist_ok=True)

    # range paths
    safe_flight_number = flight_number if flight_number is not None else "N_A"
    date_range_id = f"flight_{safe_flight_number}_{start_date_str}_to_{end_date_str}"
    date_range_base_dir = os.path.join(filtered_base_dir, date_range_id)

    output_dir_weather_filtered = os.path.join(date_range_base_dir, "processed_flights_weather")
    output_dir_cluster_data = os.path.join(date_range_base_dir, "output")
    scalers_dir_date = os.path.join(date_range_base_dir, "scalers")
    pca_dir_date = os.path.join(date_range_base_dir, "pca")
    models_dir_date = os.path.join(date_range_base_dir, "models")
    visualizations_dir_date = os.path.join(date_range_base_dir, "visualizations")

    os.makedirs(output_dir_weather_filtered, exist_ok=True)
    os.makedirs(output_dir_cluster_data, exist_ok=True)
    os.makedirs(scalers_dir_date, exist_ok=True)
    os.makedirs(pca_dir_date, exist_ok=True)
    os.makedirs(models_dir_date, exist_ok=True)
    os.makedirs(visualizations_dir_date, exist_ok=True)

    #----STEP1----
    route_processor = FlightRouteProcessor(flights_dir,skip=True) 
    route_processor.load_and_process_routes()
    route_processor.compute_average_route()
    route_processor.save_average_route('average_route_t.csv')
    avg_route_path = os.path.join(current_dir, "average_route_t.csv")


    #----STEP2----
    output_dir = os.path.join(current_dir, "flights_processed")
    data_processor = FlightDataProcessor(flights_dir, avg_route_path, output_dir)
    data_processor.process_all_flights(skip_existing=True)

    #----STEP3----
    configuration = weatherapi.Configuration()
    configuration.api_key['key'] = key
    api_instance = weatherapi.APIsApi(weatherapi.ApiClient(configuration)) # create an instance of the API class

    input_dir = os.path.join(current_dir, "flights_processed")
    output_dir = os.path.join(current_dir, "processed_flights_weather_t")
    weather_processor = FlightWeatherProcessor(input_dir, output_dir, api_instance,key)
    weather_processor.process_all_files(skip_existing=True)

    weather_processor.filtered_files(start_datetime, end_datetime, output_dir_weather_filtered, flight_number=flight_number, skip_existing=True)

    weather_processor.get_current_weather(input=avg_route_path)
    print(weather_processor.current_data)
    

    # input_dir = os.path.join(current_dir, "processed_flights_weather_t")
    # output_dir = os.path.join(current_dir, "output", "test")
    # scalers_dir = os.path.join(current_dir, "scalers", "test")
    # pca_dir = os.path.join(current_dir, "pca", "test")
    # models_dir = os.path.join(current_dir, "models", "test")
    # visualizations_dir = os.path.join(current_dir, "visualizations")

    input_dir_for_clustering = output_dir_weather_filtered

    cluster_processor = FlightClusterProcessor(input_dir_for_clustering, output_dir_cluster_data, scalers_dir_date, pca_dir_date, models_dir_date, visualizations_dir_date)
    cluster_processor.process_all_waypoints()
    cluster_processor.predict_and_save_visualizations(weather_processor.current_data)

    print("Processing completed...")
    print(len(cluster_processor.clusters))
    waypoint_dir = output_dir_cluster_data
    route_finder=RouteFinder(cluster=cluster_processor.clusters,input_dir=waypoint_dir,routes_dir=flights_dir,limit=5)
    #this contains the filename
    shortest_route=route_finder.predict_optimal_route()
    return shortest_route




# profiler = cProfile.Profile()
# profiler.enable()

# shortest_route = find_optimal_route("2024-12-01","2024-12-31")

# profiler.disable()


# stats_file_path = "profile_output.txt"
# with open(stats_file_path, "w") as f:
#     ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative') 
#     ps.print_stats()

# print(f"Profiler output saved to: {stats_file_path}")
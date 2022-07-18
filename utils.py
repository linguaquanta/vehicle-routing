import glob
import googlemaps
import numpy as np
import os
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

gmaps_key = os.environ.get('GMAPS_KEY')
gmaps = googlemaps.Client(key=gmaps_key)

def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def get_submat_from_idx_list(arr, idx_list):
    return np.array([np.take(arr[idx], idx_list) 
                       for idx in idx_list])

def drive_time(origin, destination):
    dist_mat_info = gmaps.distance_matrix(origin, destination)
    return dist_mat_info['rows'][0]['elements'][0]['duration']['value']

def drive_dist(origin, destination):
    drive = gmaps.distance_matrix(origin,destination)['rows'][0]['elements'][0]
    drive_dist = drive['distance']['value']
    return drive_dist

def compute_and_store_dist_mat(df, dist_mat_file_name):

    num_addr = len(df)
    lats = df.Latitude.tolist()
    lngs = df.Longitude.tolist()
    dist_mat = list()

    for idx_1 in range(num_addr):

        temp = list()
        origin = (lats[idx_1], lngs[idx_1])

        for idx_2 in range(num_addr):

            destination = (lats[idx_2], lngs[idx_2])
            gmaps_resp = gmaps.distance_matrix(origin, destination)['rows'][0]['elements'][0]
            # convert seconds to minutes
            cost = gmaps_resp['duration']['value']/60.0
            temp.append(cost)

        dist_mat.append(temp)

    dist_mat_cache = np.memmap(dist_mat_file_name, dtype='float64', mode='w+', shape=(num_addr, num_addr))
    dist_mat_cache[:] = np.array(dist_mat)[:]

    return dist_mat

def generate_random_coords(num_addrs, hq_file, out_file):

    # sample random addresses in Houston
    # around Rothko chapel headquarters

    # Rothko chapel coords and address
    # rothko_hq = '3900 Yupon St, Houston, TX 77006, USA'

    rothko_lat = 29.7376
    rothko_lng = -95.3962

    df_rothko = pd.DataFrame(columns=['Latitude', 'Longitude'])
    df_rothko['Latitude'] = [rothko_lat]
    df_rothko['Longitude'] = [rothko_lng]

    if hq_file not in [file for file in glob.glob("*.csv")]:
        df_rothko.to_csv(hq_file, index=False)

    # min and max lat and lng for Houston addresses
    min_lat = 29.70238 # DeBakey VA Medical Center
    max_lat = 29.813222 # Heights @ 6-10

    min_lng = -95.623658 # Eldridge @ Westheimer
    max_lng = -95.32577 # Eastwood Park

    lat_rng = max_lat-min_lat
    lng_rng = max_lng-min_lng

    # sample random points in the rectangle 
    # above according to normal dist in lat
    # and exp dist in lng

    lat_scale = 0.15
    lng_scale = 0.1

    lats = list()
    lngs = list()
    coord_pairs = list()

    r_addrs = range(num_addrs)

    for i in r_addrs:

        lat = np.random.normal(rothko_lat, scale=lat_scale*(lat_rng))
        lng = rothko_lng-np.random.exponential(scale=lng_scale*lng_rng)
        lats.append(lat)
        lngs.append(lng)
        coord_pairs.append([lat, lng])

    df_map = pd.DataFrame(columns=['Latitude', 'Longitude'])
    df_map['Latitude'] = lats
    df_map['Longitude'] = lngs
    df_map.iloc[0] = df_rothko.iloc[0]
    df_map.to_csv(out_file, index=True)

    return df_map

def generate_random_addrs(num_addrs, hq_file, out_file):

    # sample random addresses in Houston
    # around Rothko chapel headquarters

    # Rothko chapel coords and address
    # rothko_hq = '3900 Yupon St, Houston, TX 77006, USA'

    rothko_addr = '3900 Yupon St, Houston, TX 77006, USA'
    rothko_lat = 29.7376
    rothko_lng = -95.3962

    df_rothko = pd.DataFrame(columns=['Address'])
    df_rothko['Address'] = [rothko_addr]

    if hq_file not in [file for file in glob.glob("*.csv")]:
        df_rothko.to_csv(hq_file, index=False)

    # min and max lat and lng for Houston addresses
    min_lat = 29.70238 # DeBakey VA Medical Center
    max_lat = 29.813222 # Heights @ 6-10

    min_lng = -95.623658 # Eldridge @ Westheimer
    max_lng = -95.32577 # Eastwood Park

    lat_rng = max_lat-min_lat
    lng_rng = max_lng-min_lng

    # sample random points in the rectangle 
    # above according to normal dist in lat
    # and exp dist in lng

    lat_scale = 0.15
    lng_scale = 0.1

    r_addrs = range(num_addrs)
    addrs = list()

    while len(addrs) < num_addrs-1:

        lat = np.random.normal(rothko_lat, scale=lat_scale*(lat_rng))
        lng = rothko_lng-np.random.exponential(scale=lng_scale*lng_rng)
        reverse_geocode = gmaps.reverse_geocode((lat,lng))
        addr = reverse_geocode[0]['formatted_address']
        street_num = addr.split(' ')[0]
        if is_int(street_num):
            addrs.append(addr)

    df_map = pd.DataFrame(columns=['Address'])
    df_map['Address'] = addrs
    df_map.iloc[0] = df_rothko.iloc[0]

    df_map.to_csv(out_file, index=True)

    return df_map

def generate_coords_from_addrs(addrs_file):

    df_in = pd.read_csv(addrs_file, index_col=False)
    df_in = df_in.drop(columns=['Unnamed: 0'])

    addrs = df_in['Address'].tolist()
    lats = list()
    lngs = list()

    for addr in addrs:

        location = gmaps.geocode(addr)[0]['geometry']['location']
        lat = location['lat']
        lng = location['lng']
        lats.append(lat)
        lngs.append(lng)
  
    df_out = pd.DataFrame(columns=['Latitude', 'Longitude'])
    df_out['Latitude'] = lats
    df_out['Longitude'] = lngs

    return df_out

def coords_generation(n_addrs, file_coords, file_hq, regen_coords, recompute_dist_mat):

    if file_coords not in [f for f in glob.glob("*.csv")] or regen_coords:
        generate_random_coords(n_addrs, file_hq, file_coords) # does not use API calls

    df_coords = pd.read_csv(file_coords, dtype={ 'High': np.float64, 
                                                  'Low': np.float64 })
    cost_funcs = [drive_time]

    if recompute_dist_mat:
        for cost_func in cost_funcs:
            # (!) uses API calls
            compute_and_store_dist_mat(df_coords, cost_func)

    return df_coords

def google_OR_tools_mTSP_soln(dist_mat, n_vehicles):

    """Solve the CVRP problem."""
    # Instantiate the data problem.
    # `data` is a dictionary with the following keys:
    # ('dist_mat', 'num_clusters', 'depot_index')
    data = dict()
    data['distance_matrix'] = dist_mat
    data['num_vehicles'] = n_vehicles
    data['depot'] = 0

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], 
                                               data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if solution:

        max_route_distance = 0
        plan_output = list()
        route_distances = list()

        for vehicle_id in range(data['num_vehicles']):

            vehicle_plan = list()
            index = routing.Start(vehicle_id)
            route_distance = 0

            while not routing.IsEnd(index):

                vehicle_plan.append(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))


            plan_output.append(vehicle_plan)

        return plan_output


import glob
import numpy as np
import pandas as pd
from utils import (generate_coords_from_addrs,
                   compute_and_store_dist_mat,
                   google_OR_tools_mTSP_soln
                   )

def main():

    # specify file path to addresses spreadsheet
    # I am using a randomly-generated set of addresses
    # here for the sake of example
    addrs_file = 'random_addresses.csv'
    df_addrs = pd.read_csv(addrs_file, index_col=False)
    addrs = df_addrs['Address'].tolist()

    df_coords = generate_coords_from_addrs(addrs_file)
    num_coords = len(df_coords)

    dist_mat_file = 'drive_time_dist_mat.dat'
    if dist_mat_file  not in [f for f in glob.glob("*.dat")]:
        compute_and_store_dist_mat(df_coords, dist_mat_file)
    else:
        recompute_dist_mat = input(f'Recompute distance matrix? [y/n]\n')
        if recompute_dist_mat=='y':
            compute_and_store_dist_mat(df_coords, dist_mat_file)

    drive_time_mat = np.array(np.memmap(dist_mat_file, 
                                        dtype='float64', mode='r', 
                                        shape=(num_coords, num_coords)))

    num_vehicles = int(input(f'Number of vehicles: '))
    m_tsp = google_OR_tools_mTSP_soln(drive_time_mat, num_vehicles)

    cols = [f'Vehicle #{n+1}' for n in range(num_vehicles)]
    df_out = pd.DataFrame(columns=cols)

    # list of number of stops over all vehicles
    num_stops = [len(m_tsp[idx]) for idx in range(num_vehicles)]
    max_stops = max(num_stops)

    # due to unequal number of stops, need to pad
    # non-max stops columns with empty strings
    out_addrs = ['']*num_vehicles
    for idx in range(num_vehicles):

        out_addrs[idx] = list(np.take(addrs, m_tsp[idx]))

        if num_stops[idx] != max_stops:

            out_addrs[idx].extend(['']*(max_stops-num_stops[idx]))
            
        df_out[cols[idx]] = out_addrs[idx]

    vehicle_routing_solution = 'vehicle_routing_solution.csv'
    df_out.to_csv(vehicle_routing_solution)

if __name__=="__main__": 

    main() 
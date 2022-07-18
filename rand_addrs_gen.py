from utils import generate_random_addrs

hq_file = 'rothko_hq.csv'
coords_file = 'random_addresses.csv'

num_addrs = int(input(f'Number of non-HQ random addresses: '))+1
generate_random_addrs(num_addrs, hq_file, coords_file) # does not use API calls



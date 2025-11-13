import subprocess, pickle, glob, os, time

TOTAL_POSITIONS = 100
BATCH_SIZE = 10
num_batches = TOTAL_POSITIONS // BATCH_SIZE
MAX_RETRIES = 3

# run worker.py in batches with retries to get piece values
for i in range(num_batches):
    print(f'Running batch {i+1}/{num_batches}')
    for attempt in range(1, MAX_RETRIES+1):
        try:
            # capture stdout/stderr to diagnose failures
            with open(f'error_log/worker_{i}_log.txt', 'w') as log:
                subprocess.run(
                    ['python', 'worker.py', str(i), str(BATCH_SIZE)],
                    stdout=log, stderr=log, check=True)
            break  # success
        except subprocess.CalledProcessError as e:
            print(f'Batch {i} failed on attempt {attempt}')
            time.sleep(1)
            if attempt == MAX_RETRIES:
                print(f'Giving up on batch {i} after {MAX_RETRIES} attempts')

# merge results that succeeded
all_data = {}
for file in glob.glob('batch_*.pkl'):
    with open(file,'rb') as f:
        data = pickle.load(f)
    for k,v in data.items():
        all_data.setdefault(k,[]).extend(v)
    os.remove(file)

# save merged results
num = len(os.listdir('piece_values'))
with open(f'piece_values/piece_values{num}.pkl','wb') as f:
    pickle.dump(all_data,f)

print('All finished and merged')

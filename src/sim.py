import os
import time
import shutil
import argparse
import subprocess


parser = argparse.ArgumentParser(description='Time information for tidal dynamic simulation')
parser.add_argument('-c', '--case', help='Case of simulation')
parser.add_argument('-a', '--average', help='Option for turbine density averaging')
parser.add_argument('-s', '--start', help='Start date time (YYYY-MM-DDTHH:MM) of simulation')
parser.add_argument('--restart', help='Time layer for restart')
parser.add_argument('-d', '--duration', help='Duration length of simulation, number and units\ne.g. "30d", "120h", "14c" (here "c" is semi-diurnal cycle)')
parser.add_argument('-r', '--ramp-duration', help='Duration length of ramp-up, number and units\ne.g. "2d", "24", "4c" (default unit "h")')
parser.add_argument('-p', '--num-proc', help='Number of processes used in parallel simulation')
args = parser.parse_args()

# For fast debug only; in real use you need to type into command prompt
args.case = "PF1-G_test_vorticity"
args.average = "False"
args.start = "2017-08-01T00:00"
args.ramp_duration = "2d"
args.restart = ""
args.duration = "2h"
args.num_proc = "6"

if args.case:
    print('Simulation case:', args.case)
    os.environ['SIM_CASE'] = args.case

if args.start:
    print('Start simulation:', args.start)
    os.environ['SIM_START'] = args.start

if args.restart:
    print('Restart at time layer:', args.restart)
    os.environ['RESTART'] = args.restart

if args.ramp_duration:
    print('Duration of ramp-up:', args.ramp_duration)
    os.environ['RAMP_DURATION'] = args.ramp_duration

if args.duration:
    print('Duration of simulation:', args.duration)
    os.environ['SIM_DURATION'] = args.duration

if args.num_proc:
    print('Using parallel processes:', args.num_proc)
    os.environ['NUM_PROC'] = args.num_proc

if args.average:
    print('Turbine density averaging:', args.average)
    if args.average.upper() in ['TRUE', 'YES']:
        file_run = '2_run_av.py'
    elif args.average.upper() in ['FALSE', 'NO']:
        file_run = '2_run.py'
    else:
        raise ValueError('Invalid option for turbine density averaging, use "True", "Yes", "False", or "No"')

CASEDIR = f'case_{args.case}'
if not os.path.exists(f'{CASEDIR}'):
    os.makedirs(f'{CASEDIR}')
else:
    print(f'Case directory already exists: {CASEDIR}')
    print('You may want to change the case name,')
    print('or rename the existing case directory.')
    print()
    print('If you want append new output to the existing case, enter "y": ')
    if input().lower() == 'y':
        pass
    else:
        print('Exiting...')
        exit()

# Preprocessing --> Ramp-up --> Run
# print('\nPreprocessing...\n')
# with open(f'{CASEDIR}/0_preproc.log', 'w') as f_preproc_log, \
#     open(f'{CASEDIR}/0_preproc.err', 'w') as f_preproc_err:
#     subprocess.run(['python', '0_preprocessing.py'], check=True,
#                    stdout=f_preproc_log, stderr=f_preproc_err)

# print(time.ctime())
# shutil.copyfile("inputs/simulation_parameters.py", 
#                 os.path.join(CASEDIR, "simulation_parameters.py"))

# print('\nRamping-up...\n')
# with open(f'{CASEDIR}/1_ramp.log', 'w') as f_ramp_log, \
#     open(f'{CASEDIR}/1_ramp.err', 'w') as f_ramp_err:
#     subprocess.run(['mpiexec', '-np', args.num_proc, 'python', '1_ramp.py'], check=True,
#                    stdout=f_ramp_log, stderr=f_ramp_err)

print('Running...\n')
# print('\nRunning farm-bulk power for discrete turbines\n')
# file_run = '2_run_farm_pow_BAK.py'
if args.restart != "" and int(args.restart) > 0:
    print('\nRunning discrete turbines with restart (continuing prev simul)\n')
    file_run = '2_run_restart_BAK.py'

print(time.ctime())
shutil.copyfile("inputs/simulation_parameters.py",
                os.path.join(CASEDIR, "simulation_parameters.py"))

with open(f'{CASEDIR}/2_run.log', 'w') as f_run_log, \
    open(f'{CASEDIR}/2_run.err', 'w') as f_run_err:
    subprocess.run(['mpiexec', '-np', args.num_proc, 'python', file_run], check=True,
                   stdout=f_run_log, stderr=f_run_err)

print(time.ctime())

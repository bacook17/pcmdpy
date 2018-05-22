#!/bin/bash

# The abbreviated name of the script running
BASENAME="${0##*/}"

INIT_RESULTS_FILE=
FINAL_RESULTS_FILE=
STDOUT_FILE=
STDERR_FILE=

# Standard function to print an error and exit with a failing return code
error_exit () {
    echo "${BASENAME} - ${1}" >&2
    exit 1
}

# Send results and output to S3
save_to_s3() {
    if [ -f $FINAL_RESULTS_FILE ]; then
	echo "Uploading final results to s3://pcmdpy/results/${FINAL_RESULTS_FILE}"
	aws s3 cp $FINAL_RESULTS_FILE "s3://pcmdpy/results/${FINAL_RESULTS_FILE}" || echo "Unable to save results file to s3://pcmdpy/logs/${FINAL_RESULTS_FILE}"
    else
	echo "Uploading inital results to s3://pcmdpy/results/${INIT_RESULTS_FILE}"
	aws s3 cp $INIT_RESULTS_FILE "s3://pcmdpy/results/${INIT_RESULTS_FILE}" || echo "Unable to save results file to s3://pcmdpy/logs/${INIT_RESULTS_FILE}"
    fi
    echo "Uploading STDOUT logs to s3://pcmdpy/logs/${STDOUT_FILE}"
    aws s3 cp $STDOUT_FILE "s3://pcmdpy/logs/${STDOUT_FILE}" || echo "Unable to save stdout file to s3://pcmdpy/logs/${STDOUT_FILE}"
    echo "Uploading STDERR logs to s3://pcmdpy/logs/${STDERR_FILE}"
    aws s3 cp $STDERR_FILE "s3://pcmdpy/logs/${STDERR_FILE}" || echo "Unable to save stderr file to s3://pcmdpy/logs/${STDERR_FILE}"
}

# Handle external SIGINT or SIGTERM commands
exit_script() {
    echo "Received external command to quit"
    save_to_s3
    trap - SIGINT SIGTERM # clear the trap
    kill $(jobs -p)
}

trap exit_script INT TERM EXIT

# Usage info
show_help() {
error_exit "
Usage: ${BASENAME} [-h] --config-file CONFIG_FILE --run-name RUN_NAME [--data-file DATA_FILE] [--use-s3] [--clobber] [--mock]
Run pcmdpy analysis, intialized with options in CONFIG_FILE. If --data-file not provided, must be a mock run.
Results will be saved to RUN_NAME.csv, stdout copied to RUN_NAME.out, and stderr redirected to RUN_NAME.err.
Unless --clobber is given, will exit if RUN_NAME.csv, RUN_NAME.out, or RUN_NAME.err exist.

     -h / --help   display this help and exit
     --use-s3      if given, download CONFIG_FILE and DATA_FILE from AWS s3, and upload results/logs to s3
     --clobber     if given, overwrite any output files that may exist
     --mock        ignore --data-file, and assume a mock run
"
}

USE_S3=false
CLOBBER=false
MOCK_RUN=false
CONFIG_FILE=
DATA_FILE=
RUN_NAME=

# Iterate through all options
while :; do
    case $1 in
	# display usage synopsis, then exit
	-h|-\?|--help)
	    show_help
	    ;;
	# Set the configuration file. 
	-c|--config-file)
	    if [ "$2" ]; then
		CONFIG_FILE=$2
		shift
	    else
		error_exit "-c or --config-file requires a non-empty option argument."
	    fi
	    ;;
	# Set the data file
	-d|--data-file)
	    if [ "$2" ]; then
		DATA_FILE=$2
		shift
	    else
		error_exit "-d or --data-file requires a non-empty option argument."
	    fi
	    ;;
	# Set the run name.
	-r|--run-name)
	    if [ "$2" ]; then
		RUN_NAME=$2
		shift
	    else
		error_exit "-r or --run-name requires a non-empty option argument."
	    fi
	    ;;
	# Should files be downloaded/uploaded from S3?
	--use-s3)
	    USE_S3=true
	    ;;
	# Is the run a mock test?
	--mock)
	    MOCK_RUN=true
	    ;;
	# Should output files be overwritten?
	--clobber)
	    CLOBBER=true
	    ;;
	*)
	    break
    esac

    shift
done

# Check all required variables were set
if [ -z "$CONFIG_FILE" ]; then
    show_help
fi
if [ -z "$RUN_NAME" ]; then
    show_help
fi     
if [ -z "$DATA_FILE" ]; then
    if ! $MOCK_RUN; then
	show_help
    fi
fi

if $USE_S3; then
   echo "Loading files from AWS S3"
   aws s3 cp "s3://pcmdpy/config_files/${CONFIG_FILE}" $CONFIG_FILE || error_exit "Unable to find config file: s3://pcmdpy/config_files/${CONFIG_FILE}"
   if ! $MOCK_RUN; then
       aws s3 cp "s3://pcmdpy/data/${DATA_FILE}" $DATA_FILE || error_exit "Unable to find data file: s3://pcmdpy/data/${DATA_FILE}"
   fi
fi

INIT_RESULTS_FILE="${RUN_NAME}_init.csv"
FINAL_RESULTS_FILE="${RUN_NAME}_final.csv"
STDOUT_FILE="$RUN_NAME.out"
STDERR_FILE="$RUN_NAME.err"

# If clobber mode not activated, check if any output files exist
if ! $CLOBBER; then
    if [ -f $INIT_RESULTS_FILE ]; then
	error_exit "$INIT_RESULTS_FILE exists, and --clobber not activated"
    fi
    if [ -f $FINAL_RESULTS_FILE ]; then
	error_exit "$FINAL_RESULTS_FILE exists, and --clobber not activated"
    fi
    if [ -f $STDOUT_FILE ]; then
	error_exit "$STDOUT_FILE exists, and --clobber not activated"
    fi
    if [ -f $STDERR_FILE ]; then
	error_exit "$STDERR_FILE exists, and --clobber not activated"
    fi
fi

# Run pcmdpy on given config and datafile.
# Saves results to RESULTS_FILE, redirects stderr to STDERR_FILE,
# and !COPIES! stdout to STDOUT_FILE

# If a mock run
if $MOCK_RUN; then
    echo "exec: pcmd_integrate_twostage.py --config $CONFIG_FILE \
--results-init $INIT_RESULTS_FILE --results-final $FINAL_RESULTS_FILE 2> $STDERR_FILE | tee $STDOUT_FILE &"
    (pcmd_integrate_twostage.py --config $CONFIG_FILE \
		      --results-init $INIT_RESULTS_FILE --results-final $FINAL_RESULTS_FILE \
		      2> $STDERR_FILE > $STDOUT_FILE) &
    my_pid=$!
else
    echo "exec: pcmd_integrate_twostage.py --config $CONFIG_FILE \
--data $DATA_FILE --results-init $INIT_RESULTS_FILE --results-final $FINAL_RESULTS_FILE 2> $STDERR_FILE | tee $STDOUT_FILE &"
    (pcmd_integrate_twostage.py --config $CONFIG_FILE --data $DATA_FILE \
		      --results-init $INIT_RESULTS_FILE --results-final $FINAL_RESULTS_FILE \
		      2> $STDERR_FILE > $STDOUT_FILE) &
    my_pid=$!
fi

echo "PID of process: $my_pid "

# Periodically (every 2 minutes) upload results
if $USE_S3; then
    while ps -p $my_pid  # as long as the run is ongoing
    do
	save_to_s3
	tail -8 $STDOUT_FILE
	sleep 2m
    done
fi

# Either wait until done (if not using S3), or run should be done (if using S3)
wait $my_pid
CODE=$?

# Check if completed successfully
if [ $CODE -eq 0 ]; then
    echo "pcmdpy completed successfully"
else
    echo "pcmdpy failed. Error logs printed below:"
    echo "---------------------------"
    cat $STDERR_FILE
    echo "---------------------------"
fi

# Save results, stdout, and stderr regardless
if $USE_S3; then
    save_to_s3
fi

exit $CODE

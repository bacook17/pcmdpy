#!/bin/bash

# The abbreviated name of the script running
BASENAME="${0##*/}"

# Standard function to print an error and exit with a failing return code
error_exit () {
    echo "${BASENAME} - ${1}" >&2
    exit 1
}

# Usage info
show_help() {
error_exit <<EOF
Usage: ${BASENAME} [-h] --config-file CONFIG_FILE --run-name RUN_NAME [--data-file DATA_FILE] [--use-s3] [--clobber] [--mock]
Run pcmdpy analysis, intialized with options in CONFIG_FILE. If --data-file not provided, must be a mock run.
Results will be saved to RUN_NAME.csv, stdout copied to RUN_NAME.out, and stderr redirected to RUN_NAME.err.
Unless --clobber is given, will exit if RUN_NAME.csv, RUN_NAME.out, or RUN_NAME.err exist.

     -h / --help   display this help and exit
     --use-s3      if given, download CONFIG_FILE and DATA_FILE from AWS s3, and upload results/logs to s3
     --clobber     if given, overwrite any output files that may exist
     --mock        ignore --data-file, and assume a mock run
EOF
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
    error_exit
fi
if [ -z "$RUN_NAME" ]; then
    error_Exit
fi     
if [ -z "$DATA_FILE" ]; then
    if [ ! $MOCK_RUN ]; then
	error_exit
    fi
fi

if [ $USE_S3 ]; then
   echo "Loading files from AWS S3"
   aws s3 cp "s3://pcmdpy/config_files/${CONFIG_FILE}" $CONFIG_FILE || error_exit "Unable to find config file: s3://pcmdpy/config_files/${CONFIG_FILE}"
   if [ ! $MOCK_RUN ]; then
       aws s3 cp "s3://pcmdpy/data/${DATA_FILE}" $DATA_FILE || error_exit "Unable to find data file: s3://pcmdpy/data/${DATA_FILE}"
   fi
fi

RESULTS_FILE="$RUN_NAME.csv"
STDOUT_FILE="$RUN_NAME.out"
STDERR_FILE="$RUN_NAME.err"

# If clobber mode not activated, check if any output files exist
if [ ! $CLOBBER ]; then
    if [ -f $RESULTS_FILE ]; then
	error_exit "$RESULTS_FILE exists, and --clobber not activated"
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
if [ $MOCK_RUN ]; then
    python pcmdpy/pcmd_integrate.py --config $CONFIG_FILE \
	   --results $RESULTS_FILE 2> $STDERR_FILE | tee $STDOUT_FILE
else
    python pcmdpy/pcmd_integrate.py --config $CONFIG_FILE --data $DATA_FILE \
	   --results $RESULTS_FILE 2> $STDERR_FILE | tee $STDOUT_FILE
fi

# Check if completed successfully
if [ $? -eq 0 ]; then
    echo "pcmdpy completed successfully"
    if [ $USE_S3 ]; then
	echo "Uploading results to s3://pcmdpy/results/${RESULTS_FILE}"
	aws s3 cp $RESULTS_FILE "s3://pcmdpy/results/${RESULTS_FILE}" || echo "Unable to save results file to s3://pcmdpy/logs/${RESULTS_FILE}"
    fi
else
    echo "pcmdpy failed. Error logs printed below:"
    echo "---------------------------"
    cat < $ERR_FILE
fi

# Save stdout and stderr regardless
if [ $USE_S3 ]; then
    echo "Uploading STDOUT logs to s3://pcmdpy/logs/${STDOUT_FILE}"
    aws s3 cp $STDOUT_FILE "s3://pcmdpy/logs/${STDOUT_FILE}" || error_exit "Unable to save stdout file to s3://pcmdpy/logs/${STDOUT_FILE}"
    echo "Uploading STDERR logs to s3://pcmdpy/logs/${STDERR_FILE}"
    aws s3 cp $STDERR_FILE "s3://pcmdpy/logs/${STDERR_FILE}" || error_exit "Unable to save stderr file to s3://pcmdpy/logs/${STDERR_FILE}"

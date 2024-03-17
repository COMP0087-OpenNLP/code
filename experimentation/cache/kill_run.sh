#!/bin/bash

# Default process name or PID
DEFAULT_PROCESS="run.sh"

# Function to kill a process and all its children
killtree() {
    local _pid=$1
    local _sig=${2:-TERM}
    for _child in $(ps -o pid --no-headers --ppid ${_pid}); do
        killtree "${_child}" "${_sig}"
    done
    echo "Killing PID: ${_pid} with signal: ${_sig}"
    kill -"${_sig}" "${_pid}"
}

# Function to get PID from process name
get_pid_by_name() {
    local _name=$1
    pgrep -f "$_name"
}

# Main script
arg=${1:-$DEFAULT_PROCESS}
if [[ $arg =~ ^[0-9]+$ ]]; then
    # Argument is a PID
    pid=$arg
else
    # Argument is a process name, get its PID
    pid=$(get_pid_by_name "$arg")
    if [ -z "$pid" ]; then
        echo "No process found with name: $arg"
        exit 1
    fi
fi

echo "Killing process and its sub-processes: PID $pid"
killtree "$pid"

exit 0

# Note: use htop to see resource usage

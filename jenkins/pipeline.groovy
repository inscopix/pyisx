/// Common functions for build pipelines

def run_command(command, os) {
    // Run a command on a particular OS
    // On Windows, the command needs to be run as a Bat script
    // which calls the git bash shell. This was the best way to
    // run Unix commands on Windows using Jenkins
    if (os == "linux" || os == "mac") {
        sh command
    }
    else {
        bat ("sh -c \"${command}\"")
    }
}

def run(os, deploy = false) {
    stage("Setup") {
        run_command("make setup REMOTE_DIR=${IDPS_REMOTE_EXT_DIR} REMOTE_LOCAL_DIR=${IDPS_REMOTE_EXT_COPY_DIR}", os)
    }
    
    stage("Build") {
        run_command("make build THIRD_PARTY_DIR=${IDPS_REMOTE_EXT_COPY_DIR}", os)
    }

    stage("Test") {
        run_command("make test TEST_DATA_DIR=${IDPS_REMOTE_EXT_COPY_DIR}/test_data_structured", os)
    }
}

return this

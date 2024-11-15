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

def run(os, python_version, deploy = false) {
    run_command("make clean", os)
    run_command("make env PYTHON_VERSION=${python_version}", os)
    run_command("make build THIRD_PARTY_DIR=${IDPS_REMOTE_EXT_COPY_DIR} PYTHON_VERSION=${python_version}", os)
    run_command("make test TEST_DATA_DIR=${IDPS_REMOTE_EXT_COPY_DIR}/test_data_structured PYTHON_VERSION=${python_version}", os)
}

def run_all(os, deploy = false) {
    // The Jenkinsfile for the MacOS Development Builds pipeline.

    run_command("git config -f .gitmodules submodule.isxcore.url https://github.com/inscopix/isxcore.git")
    run_command("git config -f .git/config submodule.isxcore.url https://github.com/inscopix/isxcore.git")
    run_command("git submodule sync")
    run_command("git submodule update --init --remote")
    
    stage("Setup") {
        run_command("make setup REMOTE_DIR=${IDPS_REMOTE_EXT_DIR} REMOTE_LOCAL_DIR=${IDPS_REMOTE_EXT_COPY_DIR}", os)
    }

    python_versions = ["3.9", "3.10", "3.11", "3.12"]
    python_versions.each() {
        stage("Python ${it}") {
            run(os, it, deploy)
        }
    }
}

return this

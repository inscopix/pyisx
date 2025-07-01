SHELL:=/bin/bash
.PHONY: check_os build rebuild test docs

# Build paths
BUILD_DIR_ROOT=build
BUILD_DIR_MODULES=modules
BUILD_TYPE=Release
BUILD_DIR_CMAKE=cmake
BUILD_DIR_BIN=bin
BUILD_PATH=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/$(BUILD_DIR_CMAKE)
BUILD_PATH_BIN=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/$(BUILD_DIR_BIN)

# Test paths
API_TEST_RESULTS_PATH=$(PWD)/apiTestResults.xml
PYTHON_TEST_DIR=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/bin/isx

# Check for test data dir
ifndef TEST_DATA_DIR
	TEST_DATA_DIR=test_data
endif

# Check for third party dir
ifndef THIRD_PARTY_DIR
	THIRD_PARTY_DIR=third_party
endif

# Virtual environment vars
ifndef VENV_NAME
	VENV_NAME=venv
endif

# Detect OS
ifeq ($(OS), Windows_NT)
	DETECTED_OS = windows
	VENV_ACTIVATE = source ${VENV_NAME}/Scripts/activate
else
	VENV_ACTIVATE = source ${VENV_NAME}/bin/activate
	UNAME_S = $(shell uname -s)
	ifeq ($(UNAME_S), Linux)
		DETECTED_OS = linux
	else ifeq ($(UNAME_S), Darwin)
		DETECTED_OS = mac
	endif
endif

ifndef PYTHON
	PYTHON=python
endif

# Check if the directory exists using wildcard and conditional
ifeq ($(wildcard $(VENV_NAME)/.),)
  # Directory does not exist
  PYTHON_VERSION=$(shell ${PYTHON} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
else
  # Directory exists
  PYTHON_VERSION=$(shell ${VENV_ACTIVATE} && ${PYTHON} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
endif

# # Extract python version
# ifndef PYTHON_VERSION
# 	PYTHON_VERSION=$(shell ${PYTHON} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
# endif

# Set the macOS deployment version based on python version
ifeq ($(DETECTED_OS), mac)
	ifeq ($(PYTHON_VERSION), 3.9)
		_MACOSX_DEPLOYMENT_TARGET=10.11
	else ifeq ($(PYTHON_VERSION), 3.10)
		_MACOSX_DEPLOYMENT_TARGET=10.11
	else ifeq ($(PYTHON_VERSION), 3.11)
		_MACOSX_DEPLOYMENT_TARGET=10.11
	else ifeq ($(PYTHON_VERSION), 3.12)
		_MACOSX_DEPLOYMENT_TARGET=10.15
	else ifeq ($(PYTHON_VERSION), 3.13)
		_MACOSX_DEPLOYMENT_TARGET=10.15
	endif
endif

# Build flags for isxcore
VERSION_MAJOR=2
VERSION_MINOR=0
VERSION_PATCH=1
VERSION_BUILD=0
IS_BETA=1
ASYNC_API=1
WITH_ALGOS=0

# Construct cmake options
CMAKE_OPTIONS=\
    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)\
    -DISX_VERSION_MAJOR=${VERSION_MAJOR}\
    -DISX_VERSION_MINOR=${VERSION_MINOR}\
    -DISX_VERSION_PATCH=${VERSION_PATCH}\
    -DISX_VERSION_BUILD=${VERSION_BUILD}\
    -DISX_IS_BETA=${IS_BETA}\
	-DISX_ASYNC_API=${ASYNC_API} \
	-DISX_WITH_ALGOS=${WITH_ALGOS} \

# Define cmake generator based on OS
ifeq ($(DETECTED_OS), windows)
	CMAKE_GENERATOR = Visual Studio 14 2015 Win64
else ifeq ($(DETECTED_OS), linux)
	CMAKE_GENERATOR = Unix Makefiles
	CMAKE_OPTIONS += -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
else ifeq ($(DETECTED_OS), mac)
	CMAKE_GENERATOR = Xcode
endif

# ifeq ($(DETECTED_OS), windows)
# else
# 	VENV_ACTIVATE = source ${VENV_NAME}/bin/activate
# endif

ifndef BUILD_API
	BUILD_API=0
endif

check_os:
	@echo "Verifying detected OS"
ifndef DETECTED_OS
	@echo "Failed to detect supported OS"; exit 1
else
	@echo "Detected OS: ${DETECTED_OS}"
endif
ifeq ($(DETECTED_OS), mac)
	@echo "Detected python version: ${PYTHON_VERSION}, using mac osx deployment target: ${MACOSX_DEPLOYMENT_TARGET}"
endif

clean:
	@rm -rf build
	@rm -rf docs/build
	@rm -rf wheelhouse
	@rm -rf ${VENV_NAME}

setup:
	./scripts/setup -v --src ${REMOTE_DIR} --dst ${REMOTE_LOCAL_DIR} --remote-copy

# ifeq ($(DETECTED_OS), mac)
# env:
# 	CONDA_SUBDIR=osx-64 conda create -y -n $(VENV_NAME) python=$(PYTHON_VERSION) && \
# 	$(VENV_ACTIVATE) $(VENV_NAME) && \
# 	conda config --env --set subdir osx-64 && \
# 	python -m pip install build
# else
# env:
# 	conda create -y -n $(VENV_NAME) python=$(PYTHON_VERSION) && \
# 	$(VENV_ACTIVATE) $(VENV_NAME) && \
# 	python -m pip install build
# endif

ifeq ($(DETECTED_OS), mac)
env:
	${PYTHON} -m venv ${VENV_NAME}
	$(VENV_ACTIVATE) && python -m pip install '.[build,test,docs,deploy]'
else
	sh -c "${PYTHON} -m venv ${VENV_NAME}"
	$(VENV_ACTIVATE) && python -m pip install '.[build,test,docs,deploy]'
endif

ifeq ($(DETECTED_OS), mac)
build: export MACOSX_DEPLOYMENT_TARGET=${_MACOSX_DEPLOYMENT_TARGET}
endif 
build: check_os
	mkdir -p $(BUILD_PATH) && \
	cd $(BUILD_PATH) && \
	THIRD_PARTY_DIR=$(THIRD_PARTY_DIR) cmake $(CMAKE_OPTIONS) -G "$(CMAKE_GENERATOR)" ../../../
ifeq ($(DETECTED_OS), windows)
	cd $(BUILD_PATH) && \
	"/c/Program Files (x86)/MSBuild/14.0/Bin/MSBuild.exe" isx.sln //p:Configuration=$(BUILD_TYPE) //maxcpucount:8
else ifeq ($(DETECTED_OS), linux)
	cd $(BUILD_PATH) && \
	make -j2
else ifeq ($(DETECTED_OS), mac)
	cd $(BUILD_PATH) && \
	xcodebuild -alltargets -configuration $(BUILD_TYPE) -project isx.xcodeproj CODE_SIGN_IDENTITY=""
endif
	$(VENV_ACTIVATE) && \
	cd $(BUILD_PATH_BIN) && \
	python -m build

rebuild: clean build

install:
	$(VENV_ACTIVATE) && \
	pip install --force-reinstall --no-deps '$(shell ls $(BUILD_PATH_BIN)/dist/isx-*.whl)'

test: install
	$(VENV_ACTIVATE) && \
	cd build/Release && \
	ISX_TEST_DATA_PATH='$(shell realpath $(TEST_DATA_DIR))' python -m pytest --disable-warnings -v -s --junit-xml=$(API_TEST_RESULTS_PATH) test $(TEST_ARGS)

ifeq ($(BUILD_API), 1)
docs: install
endif
docs:
	$(VENV_ACTIVATE) && \
	sphinx-build docs docs/build

# Used for fixing linux wheel installs before deploying to pypi
repair-linux:
	docker run \
		-v $(shell pwd):/io \
		-u $(shell id -u ${USER}):$(shell id -g ${USER}) \
		quay.io/pypa/manylinux_2_34_x86_64 \
		/bin/bash -c "cd /io && LD_LIBRARY_PATH=/io/build/Release/bin/isx/lib:$LD_LIBRARY_PATH auditwheel repair /io/build/Release/bin/dist/isx*.whl"

ifeq ($(DETECTED_OS), linux)
deploy: repair-linux
	$(VENV_ACTIVATE) && \
	twine upload '$(shell ls wheelhouse/isx-*.whl)'
else
deploy:
	$(VENV_ACTIVATE) && \
	twine upload '$(shell ls $(BUILD_PATH_BIN)/dist/isx-*.whl)'
endif

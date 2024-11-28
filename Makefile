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

# Extract python version
ifndef PYTHON_VERSION
	PYTHON_VERSION=$(shell python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
endif

# Detect OS
ifeq ($(OS), Windows_NT)
	DETECTED_OS = windows
else
	UNAME_S = $(shell uname -s)
	ifeq ($(UNAME_S), Linux)
		DETECTED_OS = linux
	else ifeq ($(UNAME_S), Darwin)
		DETECTED_OS = mac
		
# Set the macOS deployment version based on python version
		ifeq ($(PYTHON_VERSION), 3.9)
			_MACOSX_DEPLOYMENT_TARGET=10.11
		else ifeq ($(PYTHON_VERSION), 3.10)
			_MACOSX_DEPLOYMENT_TARGET=10.11
		else ifeq ($(PYTHON_VERSION), 3.11)
			_MACOSX_DEPLOYMENT_TARGET=10.11
		else ifeq ($(PYTHON_VERSION), 3.12)
			_MACOSX_DEPLOYMENT_TARGET=10.15
		endif
	endif
endif

# Build flags for isxcore
VERSION_MAJOR=2
VERSION_MINOR=0
VERSION_PATCH=1
VERSION_BUILD=0
IS_BETA=1
WITH_CUDA=0
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
    -DISX_WITH_CUDA=${WITH_CUDA}\
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

# Virtual environment vars
ifndef VENV_NAME
	VENV_NAME=pyisx
endif

# Define cmake generator based on OS
ifeq ($(DETECTED_OS), windows)
	VENV_ACTIVATE = source '$(shell conda info --base)/Scripts/activate'
else
	VENV_ACTIVATE = source $(shell conda info --base)/bin/activate
endif

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

setup:
	./scripts/setup -v --src ${REMOTE_DIR} --dst ${REMOTE_LOCAL_DIR} --remote-copy

ifeq ($(DETECTED_OS), mac)
env:
	CONDA_SUBDIR=osx-64 conda create -y -n $(VENV_NAME) python=$(PYTHON_VERSION) && \
	$(VENV_ACTIVATE) $(VENV_NAME) && \
	conda config --env --set subdir osx-64 && \
	python -m pip install build
else
env:
	conda create -y -n $(VENV_NAME) python=$(PYTHON_VERSION) && \
	$(VENV_ACTIVATE) $(VENV_NAME) && \
	python -m pip install build
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
	cd $(BUILD_PATH_BIN) && \
	$(VENV_ACTIVATE) $(VENV_NAME) && \
	python -m build

rebuild: clean build
 
test:
	$(VENV_ACTIVATE) $(VENV_NAME) && \
	pip install --force-reinstall '$(shell ls $(BUILD_PATH_BIN)/dist/isx-*.whl)[test]' && \
	cd build/Release && \
	ISX_TEST_DATA_PATH='$(shell realpath $(TEST_DATA_DIR))' python -m pytest --disable-warnings -v -s --junit-xml=$(API_TEST_RESULTS_PATH) test $(TEST_ARGS)

ifeq ($(BUILD_API), 1)
docs: build
	$(VENV_ACTIVATE) $(VENV_NAME) && \
	pip install --force-reinstall '$(shell ls $(BUILD_PATH_BIN)/dist/isx-*.whl)[docs]'
endif
docs:
	$(VENV_ACTIVATE) $(VENV_NAME) && \
	sphinx-build docs docs/build

deploy:
	$(VENV_ACTIVATE) $(VENV_NAME) && \
	pip install twine && \
	twine upload --repository testpypi '$(shell ls $(BUILD_PATH_BIN)/dist/isx-*.whl)'

.PHONY: check_os build rebuild test docs

BUILD_DIR_ROOT=build
BUILD_DIR_MODULES=modules
BUILD_TYPE=Release
BUILD_DIR_CMAKE=cmake
BUILD_DIR_BIN=bin
BUILD_PATH=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/$(BUILD_DIR_CMAKE)
BUILD_PATH_BIN=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/$(BUILD_DIR_BIN)

API_TEST_RESULTS_PATH=$(PWD)/apiTestResults.xml
PYTHON_TEST_DIR=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/bin/isx

ifndef TEST_DATA_DIR
	TEST_DATA_DIR=test_data
endif

ifndef THIRD_PARTY_DIR
	THIRD_PARTY_DIR=third_party
endif

PYTHON_VERSION=$(shell python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

ifeq ($(OS), Windows_NT)
	DETECTED_OS = windows
else
	UNAME_S = $(shell uname -s)
	ifeq ($(UNAME_S), Linux)
		DETECTED_OS = linux
	else ifeq ($(UNAME_S), Darwin)
		DETECTED_OS = mac
		
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

VERSION_MAJOR=2
VERSION_MINOR=0
VERSION_PATCH=0
VERSION_BUILD=0
IS_BETA=1
WITH_CUDA=0
ASYNC_API=1
WITH_ALGOS=0

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

ifeq ($(DETECTED_OS), windows)
	CMAKE_GENERATOR = Visual Studio 14 2015 Win64
else ifeq ($(DETECTED_OS), linux)
	CMAKE_GENERATOR = Unix Makefiles
	CMAKE_OPTIONS += -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
else ifeq ($(DETECTED_OS), mac)
	CMAKE_GENERATOR = Xcode
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
	python -m build

rebuild: clean build
 
test: build
	pip install --force-reinstall '$(shell ls $(BUILD_PATH_BIN)/dist/isx-*.whl)[test]'
	cd build/Release && \
	ISX_TEST_DATA_PATH=$(TEST_DATA_DIR) python -m pytest --disable-warnings -v -s --junit-xml=$(API_TEST_RESULTS_PATH) test $(TEST_ARGS)

docs: 
	sphinx-build docs docs/build

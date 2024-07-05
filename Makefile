.PHONY: build test

BUILD_DIR_ROOT=build
BUILD_DIR_MODULES=modules
BUILD_TYPE=Release
BUILD_DIR_CMAKE=cmake
BUILD_DIR_BIN=bin
BUILD_PATH=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/$(BUILD_DIR_CMAKE)
BUILD_PATH_BIN=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/$(BUILD_DIR_BIN)

ifndef TEST_DATA_DIR
	TEST_DATA_DIR=test_data
endif

ifndef THIRD_PARTY_DIR
	THIRD_PARTY_DIR=third_party
endif

API_TEST_RESULTS_PATH=$(PWD)/apiTestResults.xml
PYTHON_TEST_DIR=$(BUILD_DIR_ROOT)/$(BUILD_TYPE)/bin/isx

ifeq ($(OS), Windows_NT)
	DETECTED_OS = windows
else
	UNAME_S = $(shell uname -s)
	ifeq ($(UNAME_S), Linux)
		DETECTED_OS = linux
	else ifeq ($(UNAME_S), Darwin)
		DETECTED_OS = mac
	endif
endif

VERSION_MAJOR=1
VERSION_MINOR=9
VERSION_PATCH=5
VERSION_BUILD=0
IS_BETA=1
WITH_CUDA=0
IS_MINIMAL_API=1

CMAKE_OPTIONS=\
    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)\
    -DISX_VERSION_MAJOR=${VERSION_MAJOR}\
    -DISX_VERSION_MINOR=${VERSION_MINOR}\
    -DISX_VERSION_PATCH=${VERSION_PATCH}\
    -DISX_VERSION_BUILD=${VERSION_BUILD}\
    -DISX_IS_BETA=${IS_BETA}\
    -DISX_WITH_CUDA=${WITH_CUDA}\
    -DISX_IS_MINIMAL_API=${IS_MINIMAL_API}

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

clean:
	@rm -rf build

build: check_os
	@echo ${CMAKE_GENERATOR} $(BUILD_PATH) $(CMAKE_OPTIONS) $(THIRD_PARTY_DIR)
	mkdir -p $(BUILD_PATH) && \
	cd $(BUILD_PATH) && \
	THIRD_PARTY_DIR=$(THIRD_PARTY_DIR) cmake $(CMAKE_OPTIONS) -G "$(CMAKE_GENERATOR)" ../../../
ifeq ($(DETECTED_OS), windows)
	cd $(BUILD_PATH) && \
	"/c/Program Files (x86)/MSBuild/14.0/Bin/MSBuild.exe" Project.sln //p:Configuration=$(BUILD_TYPE) //maxcpucount:8
else ifeq ($(DETECTED_OS), linux)
	cd $(BUILD_PATH) && \
	make -j2
else ifeq ($(DETECTED_OS), mac)
	cd $(BUILD_PATH) && \
	xcodebuild -alltargets -configuration $(BUILD_TYPE) -project Project.xcodeproj CODE_SIGN_IDENTITY=""
endif
	cd $(BUILD_PATH_BIN) && \
	python -m build

rebuild: clean build
 
test: build
	cd $(BUILD_PATH_BIN)/dist && pip install --force-reinstall --no-deps isx-1.9.2-py3-none-any.whl
	cd build/Release && \
	ISX_TEST_DATA_PATH=$(TEST_DATA_DIR) python -m pytest --disable-warnings -v -s --junit-xml=$(API_TEST_RESULTS_PATH) test
5
cmake_minimum_required(VERSION 3.5)

set(ISX_CMAKE_TOOLS_DIR ${CMAKE_CURRENT_LIST_DIR}/isxcore/tools/cmake)
include(${ISX_CMAKE_TOOLS_DIR}/configure.cmake)
include(${ISX_CMAKE_TOOLS_DIR}/core.cmake)

#
# Public API
#
set(TARGET_NAME_PUBLIC_API "isxpublicapi")
set(PUBLIC_API_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/isxcore/wrapper/src)
set(PUBLIC_API_API_DIR ${CMAKE_CURRENT_LIST_DIR}/isxcore/wrapper/api)
set(PUBLIC_API_SRCS ${PUBLIC_API_SRC_DIR}/isxCoreC.cpp ${PUBLIC_API_SRC_DIR}/isxUtilsC.cpp)
set(PUBLIC_API_HDRS ${PUBLIC_API_SRC_DIR}/isxUtilsC.h)
set(PUBLIC_API_API_HDRS ${PUBLIC_API_API_DIR}/isxCoreC.h)

add_library(${TARGET_NAME_PUBLIC_API} SHARED ${PUBLIC_API_SRCS} ${PUBLIC_API_HDRS} ${PUBLIC_API_API_HDRS})

target_include_directories(${TARGET_NAME_PUBLIC_API} PUBLIC ${PUBLIC_API_API_DIR})
target_include_directories(${TARGET_NAME_PUBLIC_API} PRIVATE
    ${CORE_API_DIR}
    ${OPENCV_HEADER_SEARCH_PATHS}
    ${JSON_HEADER_SEARCH_PATHS})

set(API_DEST_DIR ${CMAKE_BINARY_DIR}/../bin)
set(PYTHON_API_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})
set(APP_LIB_DIR ${API_DEST_DIR}/isx/lib)

if (${ISX_OS_MACOS})
    set(QT_CORE_SHARED_LIB_FILES ${QT_DIR}/lib/QtCore.framework/Versions/5/QtCore)
    installFiles(${TARGET_NAME_PUBLIC_API} ${APP_LIB_DIR}/QtCore.framework/Versions/5 "${QT_CORE_SHARED_LIB_FILES}")
elseif (${ISX_OS_LINUX})
    set(QT_CORE_SHARED_LIB_FILES 
        ${QT_DIR}/lib/libQt5Core.so.5
        ${QT_DIR}/lib/libicui18n.so.56
        ${QT_DIR}/lib/libicuuc.so.56
        ${QT_DIR}/lib/libicudata.so.56
    )
    installFiles(${TARGET_NAME_PUBLIC_API} ${APP_LIB_DIR} "${QT_CORE_SHARED_LIB_FILES}")
elseif (${ISX_OS_WIN32})
    set(QT_CORE_SHARED_LIB_FILES 
        ${QT_DIR}/bin/Qt5Core.dll
    )
    installFiles(${TARGET_NAME_PUBLIC_API} ${APP_LIB_DIR} "${QT_CORE_SHARED_LIB_FILES}")
endif()

installFfmpegSharedLibs(${TARGET_NAME_PUBLIC_API} ${APP_LIB_DIR})

set_target_properties(${TARGET_NAME_PUBLIC_API} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${APP_LIB_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${APP_LIB_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${APP_LIB_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${APP_LIB_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${APP_LIB_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${APP_LIB_DIR}
)
setCommonCxxOptionsForTarget(${TARGET_NAME_PUBLIC_API})
setOsDefinesForTarget(${TARGET_NAME_PUBLIC_API})
disableVisualStudioWarnings(${TARGET_NAME_PUBLIC_API})

# OS-specific properties for shared lib
# mac: change output lib name to *.so (default *.dylib)
# mac & linux: set rpath
if(${ISX_OS_MACOS})
    string(APPEND SO_NAME_PUBLIC_API ${TARGET_NAME_PUBLIC_API} ".so")

    set_target_properties(${TARGET_NAME_PUBLIC_API} PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME ${SO_NAME_PUBLIC_API}
        SUFFIX ""
        #SOVERSION "1.0.0.0"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path"
    )
elseif(${ISX_OS_LINUX})
    set_target_properties(${TARGET_NAME_PUBLIC_API} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/"
        # Interposition of symbols from static libraries in MATLAB was causing
        # major problems, so we only export a few select symbols now.
        # We may want to do the same for macOS, which also exposes these
        # symbols (but hasn't caused a problem yet).
        LINK_FLAGS "-Wl,--version-script=${PUBLIC_API_API_DIR}/export_map"
    )
endif()

target_link_libraries(${TARGET_NAME_PUBLIC_API} PRIVATE
    ${TARGET_NAME_CORE}
    ${QT_CORE_LINK_LIBRARIES}
)

# Deploy Python files
set(PYTHON_API_DEST_DIR ${API_DEST_DIR}/isx)
file(COPY ${PYTHON_API_SRC_DIR}/isx/ DESTINATION ${PYTHON_API_DEST_DIR})
file(COPY ${PYTHON_API_SRC_DIR}/setup.py DESTINATION ${API_DEST_DIR})
file(COPY ${PYTHON_API_SRC_DIR}/MANIFEST.in DESTINATION ${API_DEST_DIR})
file(COPY ${PYTHON_API_SRC_DIR}/README.md DESTINATION ${API_DEST_DIR})
file(COPY ${PYTHON_API_SRC_DIR}/isx/test DESTINATION ${API_DEST_DIR}/../)

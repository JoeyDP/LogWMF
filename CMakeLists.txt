cmake_minimum_required(VERSION 3.22)
project(LogMF_Experiments)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS "-O3 -Wall -march=native")


include_directories(lib/eigen-3.3.9/)
include_directories(src)

add_executable(ials
        src/recommender.cpp
        src/ials.cpp
        src/ials_main.cpp
)
target_link_libraries(ials PRIVATE Threads::Threads)

add_executable(ialspp
        src/recommender.cpp
        src/ialspp.cpp
        src/ialspp_main.cpp
)
target_link_libraries(ialspp PRIVATE Threads::Threads)


add_executable(logmf
        src/recommender.cpp
        src/ials.cpp
        src/logmf.cpp
        src/logmf_main.cpp
)
target_link_libraries(logmf PRIVATE Threads::Threads)

add_executable(logmfpp
        src/recommender.cpp
        src/ialspp.cpp
        src/logmfpp.cpp
        src/logmfpp_main.cpp
)
target_link_libraries(logmfpp PRIVATE Threads::Threads)

add_executable(ease
        src/recommender.cpp
        src/ease.cpp
        src/ease_main.cpp
)
target_link_libraries(ease PRIVATE Threads::Threads)

add_executable(easepp
        src/recommender.cpp
        src/easepp.cpp
        src/easepp_main.cpp
)
target_link_libraries(easepp PRIVATE Threads::Threads)

add_executable(logeasepp
        src/recommender.cpp
        src/ialspp.cpp
        src/logeasepp.cpp
        src/logeasepp_main.cpp
)
target_link_libraries(logeasepp PRIVATE Threads::Threads)


add_executable(mf
        src/recommender.cpp
        src/mf.cpp
        src/mf_main.cpp
)
target_link_libraries(mf PRIVATE Threads::Threads)

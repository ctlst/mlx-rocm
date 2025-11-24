target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/add.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/subtract.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/multiply.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/divide.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/remainder.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/maximum.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/minimum.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/power.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/equal.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/not_equal.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/greater.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/greater_equal.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/less.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/less_equal.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/logical_and.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/logical_or.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/bitwise_and.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/bitwise_or.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/bitwise_xor.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/left_shift.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/right_shift.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/arctan2.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/log_add_exp.hip)


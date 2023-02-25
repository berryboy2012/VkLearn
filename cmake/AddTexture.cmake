cmake_minimum_required(VERSION 3.22)
function(add_texture TARGET TEXTURE)
    # find_program(TEXTPROG texture_processor)

    set(current-texture-path ${CMAKE_CURRENT_SOURCE_DIR}/textures/${TEXTURE})
    set(current-output-path ${CMAKE_BINARY_DIR}/textures/${TEXTURE})

    # Add a custom command for possible processing.
    get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})
    add_custom_command(
            OUTPUT ${current-output-path}
            #COMMAND ${TEXTPROG} -args ${current-output-path} ${current-texture-path}
            COMMAND ${CMAKE_COMMAND} -E copy ${current-texture-path} ${current-output-path}
            DEPENDS ${current-texture-path}
            IMPLICIT_DEPENDS CXX ${current-texture-path}
            VERBATIM)

    # Make sure our build depends on this output.
    set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${current-output-path})
endfunction(add_texture)
### OFFSETS ON GLRM
function add_offset!(glrm::AbstractGLRM)
    glrm.rx, glrm.ry = lastentry1(glrm.rx), lastentry_unpenalized(glrm.ry)
    return glrm
end

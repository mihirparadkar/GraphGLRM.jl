### OFFSETS ON GLRM
function add_offset!(glrm::AbstractGLRM)
    glrm.rx, glrm.ry = lastentry1(glrm.rx), lastentry_unpenalized(glrm.ry)
    return glrm
end

function equilibrate_variance!(glrm::AbstractGLRM, columns_to_scale = 1:size(glrm.A,2))
  for i in columns_to_scale
    nomissing = glrm.A[glrm.observed_examples[i],i]
    if length(nomissing)>1
      varlossi = avgerror(glrm.losses[i], nomissing)
    else
      varlossi = 1
    end
    if varlossi > 0
      # rescale the losses and regularizers for each column by the inverse of the empirical variance
      scale!(glrm.losses[i], scale(glrm.losses[i])/varlossi)
    end
  end
  glrm
end

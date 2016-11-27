

### SCALINGS AND OFFSETS ON GLRM
function LowRankModels.add_offset!(glrm::GGLRM)
    glrm.rx = lastentry1(glrm.rx)
    glrm.ry = Dict(lastentry_unpenalized(r) => i for (r,i) in glrm.ry)
    return glrm
end

## equilibrate variance
# scale all columns inversely proportional to mean value of loss function
# makes sense when all loss functions used are nonnegative
function LowRankModels.equilibrate_variance!(glrm::GGLRM)
    for i in 1:size(glrm.A,2)
        nomissing = glrm.A[glrm.observed_examples[i],i]
        if length(nomissing)>0
            varlossi = avgerror(glrm.losses[i], nomissing)
        else
            varlossi = 1
        end
        if varlossi > 0
            # rescale the losses and regularizers for each column by the inverse of the empirical variance
            scale!(glrm.losses[i], scale(glrm.losses[i])/varlossi)
        end
    end
    for r in keys(glrm.ry)
      if !isa(r, MatrixRegularizer)
        nomissing = glrm.A[glrm.observed_examples[glrm.ry[r]],glrm.ry[r]]
        varregi = var(nomissing) # TODO make this depend on the kind of regularization; this assumes QuadLoss
        if varregi > 0
          scale!(r, scale(r)/varregi)
        end
      else
        varregi = 1
        scale!(r, scale(r)/varregi)
      end
    end
    return glrm
end

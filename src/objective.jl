function xrowpenalty(glrm::GGLRM, i::Int)
  evaluate(glrm.rx, view(glrm.X, :, i))
end

#helper function for calculating regularization penalty on X
function xpenalty(glrm::GGLRM)

  #If x regularizer is whole-matrix, then hit this codepath
  if (isa(glrm.rx, lastentry1) && isa(glrm.rx.r, MatrixRegularizer)) || isa(glrm.rx, MatrixRegularizer)
    return evaluate(glrm.rx, glrm.X)
  else
    penalty = 0.
    for i in 1:size(glrm.X,2)
      penalty += xrowpenalty(glrm, i)
    end
  end
  penalty
end

function ycolpenalty(glrm::GGLRM, r::Regularizer)
  yidxs = get_yidxs(glrm.losses)
  #Either it's a single-column regularizer, so do the normal process
  if isa(glrm.ry[r], Int)
    return evaluate(r, view(glrm.Y, :, yidxs[glrm.ry[r]]))
  #Or it's a matrix regularizer so evaluate on multiple columns. They are guaranteed to not overlap
  else
    return evaluate(r, view(glrm.Y, :, glrm.ry[r]))
  end
end

function ypenalty(glrm::GGLRM)
  penalty = 0.
  for r in keys(glrm.ry)
    penalty += ycolpenalty(glrm, r)
  end
  penalty
end

function col_objective(glrm::GGLRM, r::Regularizer, newY::Matrix{Float64})
  yidxs = get_yidxs(glrm.losses)
  j = glrm.ry[r]
  obj = 0.
  #It's a number so proceed as normal
  if isa(j, Int)
    XY = glrm.X'*view(newY, :, yidxs[j])

    #Now check dimension of XY, since single-dimension losses give 1-d vector
    if isa(XY, Vector{Float64})
      for i in glrm.observed_examples[j]
        obj += evaluate(glrm.losses[j], XY[i], glrm.A[i,j])
      end
    else
      for i in glrm.observed_examples[j]
        #Every column is from yidxs and j is a number
        obj += evaluate(glrm.losses[j], XY[i, :], glrm.A[i,j])
      end
    end
  #Figure out the offsets and evaluate the loss on those columns
  else
    colids = [yidxs[ind] for ind in j]
    XY = glrm.X'*view(newY, :, colids)
    for (idx,f) in enumerate(j)
      for i in glrm.observed_examples[f]
        #Use the idx-th column with the corresponding column of A
        obj += evaluate(glrm.losses[f], XY[i,idx], glrm.A[i,f])
      end
    end
  end
  obj + ycolpenalty(glrm,r)
end

#whole regularization penalty of a GGLRM
function whole_penalty(glrm::GGLRM)
  xpenalty(glrm) + ypenalty(glrm)
end

### Loss functions for whole reconstruction matrix
function whole_losses(glrm::GGLRM, XY::Array{Float64,2};
                   yidxs = get_yidxs(glrm.losses)) # mapping from columns of A to columns of Y; by default, the identity
  m,n = size(glrm.A)
  @assert(size(XY)==(m,yidxs[end][end]))
  err = 0.0
  for j=1:n
      for i in glrm.observed_examples[j]
          err += evaluate(glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
      end
  end
  err
end

function whole_objective(glrm::GGLRM, XY::Matrix{Float64})
  whole_losses(glrm, XY) + whole_penalty(glrm)
end

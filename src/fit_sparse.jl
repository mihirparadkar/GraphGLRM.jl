function reconstruct_obs!(g::GGLRM, XY::SparseMatrixCSC{Float64}; X = g.X, Y = g.Y)
  yidxs = get_yidxs(g.losses)
  obsex = g.observed_examples
  for j in 1:length(g.losses)
    @inbounds Yj = view(Y, :, yidxs[j])
    for i in obsex[j]
      Xi = view(X, :, i)
      if isa(yidxs[j], Number)
        XY[i, yidxs[j]] = (Xi'Yj)[1]
      else
        XY[i, yidxs[j]] = Xi'Yj
      end
    end
  end
end

function reconstruct_obs(g::GGLRM)
  XY = spzeros(size(g.X,2), size(g.Y,2))
  reconstruct_obs!(g, XY)
  XY
end

#Does a line search for the step size for X, returns the new step size
@inline function _threadedproxStepX!(g::AbstractGLRM, params::ProxGradParams,
                            newX::Matrix{Float64}, gx::Matrix{Float64},
                            XY::SparseMatrixCSC{Float64}, newXY::SparseMatrixCSC{Float64},
                            αx::Number)
  #l = 1.5
  l = maximum(map(length, g.observed_features))+1#(mapreduce(length,+,g.observed_features) + 1)

  obj = threaded_loss_objective(g,XY) + evaluate(g.rx, g.X)
  newobj = NaN
  while αx > params.min_stepsize #Linesearch to find the new step size
    stepsize = αx/l

    axpy!(-stepsize, gx, newX)
    prox!(g.rx, newX, stepsize)
    reconstruct_obs!(g, newXY, X=newX)#At_mul_B!(newXY, newX, g.Y)
    newobj = threaded_loss_objective(g, newXY) + evaluate(g.rx, newX)
    #newobj = threaded_objective(g, newXY)
    if newobj < obj
      copy!(g.X, newX)
      αx *= 1.03#(1.05)
      break
    else #Try again with smaller step-size
      copy!(newX, g.X)
      αx *= 0.9 #0.7
      if αx < params.min_stepsize
        αx = params.min_stepsize * 1.1
        break
      end #if
    end #if else
  end #while
  αx, newobj
end

@inline function _threadedproxStepY!(g::AbstractGLRM, params::ProxGradParams,
                              newY::Matrix{Float64}, gy::Matrix{Float64},
                              XY::SparseMatrixCSC{Float64}, newXY::SparseMatrixCSC{Float64},
                              αy::Number)
  #l = 1.5
  l = maximum(map(length, g.observed_examples)) + 1#(mapreduce(length,+,g.observed_features) + 1)
  #obj = threaded_objective(g,XY)
  obj = threaded_loss_objective(g, XY) + evaluate(g.ry, g.Y)
  newobj = NaN
  while αy > params.min_stepsize #Linesearch to find the new step size
    stepsize = αy/l
    axpy!(-stepsize, gy, newY)
    prox!(g.ry, newY, stepsize)
    reconstruct_obs!(g, newXY, Y=newY)#At_mul_B!(newXY, g.X, newY)
    newobj = threaded_loss_objective(g, newXY) + evaluate(g.ry, newY)
    #newobj = threaded_objective(g, newXY)
    if newobj < obj
      copy!(g.Y, newY)
      αy *= 1.03#1.05
      break
    else #Try again with smaller step-size
      copy!(newY, g.Y)
      αy *= 0.9 #0.7
      if αy < params.min_stepsize
        αy = params.min_stepsize * 1.1
        break
      end #if
    end #if else
  end #while
  αy, newobj
end

function fit_sparse!(g::GGLRM,
                      params::ProxGradParams=ProxGradParams(),
                      ch::ConvergenceHistory=ConvergenceHistory("ProxGradGLRM"),
                      verbose=true)
  X,Y = g.X, g.Y
  A = g.A
  losses, rx, ry = g.losses, g.rx, g.ry
  yidxs = get_yidxs(g.losses)

  #Initialize X*Y
  XY = spzeros(size(X,2), size(Y,2))
  reconstruct_obs!(g, XY)

  tm = 0
  update_ch!(ch, tm, whole_objective(g,XY))

  #Step sizes
  αx = params.stepsize
  αy = params.stepsize

  # stopping criterion: stop when decrease in objective < tol, scaled by the number of observations
  scaled_abs_tol = params.abs_tol * mapreduce(length,+,g.observed_features)

  newX = copy(X)
  newY = copy(Y)
  newXY = copy(XY)

  #Gradient matrices
  gx = zeros(X)
  gy = zeros(Y)

  if verbose println("Fitting GGLRM") end
  for t in 1:params.max_iter
    #X update-----------------------------------------------------------------
    #_threadedupdateGradX!(g,XY,gx)
    _threadedupdateGradX!(g,XY,gx)
    #Take a prox step with line search
    αx, objx = _threadedproxStepX!(g, params, newX, gx, XY, newXY, αx)
    reconstruct_obs!(g, XY) #Get the new XY matrix for objective

    #Y Update---------------------------------------------------------------
    #_threadedupdateGradY!(g,XY,gy)
    _threadedupdateGradY!(g,XY,gy)
    αy, objy = _threadedproxStepY!(g, params, newY, gy, XY, newXY, αy)
    reconstruct_obs!(g, XY) #Get the new XY matrix for objective
    if t % 10 == 0
      if verbose
        println("Iteration $t, objective value: $(objy + evaluate(rx, g.X))")
      end
    end
    #Update convergence history
    obj = objy + evaluate(rx, g.X)
    tm = time() - tm
    update_ch!(ch, tm, obj)
    tm = time()
    #Check stopping criterion
    obj_decrease = ch.objective[end-1] - obj
    if t>10 && (obj_decrease < scaled_abs_tol || obj_decrease/obj < params.rel_tol)
      if verbose
        println("Iteration $t, objective value: $obj")
      end
      break
    end
  end #For
  ch
end

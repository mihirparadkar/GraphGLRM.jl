function reconstruct_obs!(g::GGLRM, XY::Array{VecOrMat{Float64}}; X = g.X, Y = g.Y)
  yidxs = get_yidxs(g.losses)
  obsex = g.observed_examples
  for j in 1:length(g.losses)
    @inbounds Yj = Y[:,yidxs[j]]
    @inbounds Xisj = X[:, obsex[j]]
    #Is a column of the XY matrix proper
    At_mul_B!(XY[j], Xisj, Yj)
  end
end

function reconstruct_obs(g::GGLRM)
  yidxs = get_yidxs(g.losses)
  obsex = g.observed_examples
  XY = VecOrMat{Float64}[zeros(length(obsex[j]), length(yidxs[j]))
                                        for j in 1:length(obsex)]
  for j in 1:length(XY)
    if size(XY[j], 2) == 1
      XY[j] = vec(XY[j])
    end
  end
  reconstruct_obs!(g, XY)
  XY
end

#Evaluates the loss function over all the observed values, optimized for few observations
function sparse_loss_objective(g::GGLRM, XY::Array{VecOrMat{Float64}})
  yidxs = get_yidxs(g.losses)
  obj = zeros(Threads.nthreads())
  Threads.@threads for j in 1:length(g.losses)
    obsex = g.observed_examples[j]
    #A may be a DataFrame or DataArray, but native Arrays are fastest
    @inbounds Aj = convert(Array, g.A[obsex, j])
    #This assumes a reconstruction over only the observations as done in reconstruct_obs!
    @inbounds XYj = XY[j]
    obj[Threads.threadid()] += evaluate(g.losses[j], XYj, Aj)
  end
  sum(obj)
end

#Calculates the whole objective of the GLRM
function sparse_whole_objective(g::GGLRM, XY::Array{VecOrMat{Float64}};
                        X::Matrix{Float64}=g.X, Y::Matrix{Float64}=g.Y)
  sparse_loss_objective(g, XY) + evaluate(g.rx, X) + evaluate(g.ry, Y)
end

#Makes some performance optimizations
#Finds all of the gradients in a column so that the size of the column gradient is consistent
#as opposed to the row gradient which has column-chunks and whatnot
@inline function _threadedupdateGradX!(g::AbstractGLRM, XY::Array{VecOrMat{Float64}}, gx::Matrix{Float64})
  yidxs = get_yidxs(g.losses)
  scale!(gx,0)

  #Update the gradient, go by column then by row
  Threads.@threads for j in 1:length(g.losses)
    #Yj for computing gradient
    @inbounds Yj = view(g.Y, :, yidxs[j])
    obsex = g.observed_examples[j]

    #Take whole columns of XY and A and take the gradient of those
    @inbounds Aj = convert(Array, g.A[obsex, j])
    @inbounds XYj = XY[j]
    grads = grad(g.losses[j], XYj, Aj)

    #Single dimensional losses
    if isa(grads, Vector)
      for e in 1:length(obsex)
        #i = obsex[e], so update that portion of gx
        @inbounds gxi = view(gx, :, obsex[e])
        axpy!(grads[e], Yj, gxi)
      end
    else
      for e in 1:length(obsex)
        @inbounds gxi = view(gx, :, obsex[e])
        gemm!('N','N',1.0,Yj, grads[e,:], 1.0, gxi)
      end
    end
  end
end

@inline function _threadedupdateGradY!(g::AbstractGLRM, XY::Array{VecOrMat{Float64}}, gy::Matrix{Float64})
  yidxs = get_yidxs(g.losses)
  #scale y gradient to zero
  scale!(gy, 0)

  #Update the gradient
  Threads.@threads for j in 1:length(g.losses)
    @inbounds gyj = view(gy, :, yidxs[j])
    obsex = g.observed_examples[j]
    #Take whole columns of XY and A and take the gradient of those
    @inbounds Aj = convert(Array, g.A[obsex, j])
    @inbounds XYj = XY[j]
    grads = grad(g.losses[j], XYj, Aj)
    #Single dimensional losses
    if isa(grads, Vector)
      for e in 1:length(obsex)
        #i = obsex[e], so use that for Xi
        @inbounds Xi = view(g.X, :, obsex[e])
        axpy!(grads[e], Xi, gyj)
      end
    else
      for e in 1:length(obsex)
        @inbounds Xi = view(g.X, :, obsex[e])
        gemm!('N','T',1.0, Xi, grads[e,:], 1.0, gyj)
      end
    end
  end
end

#Does a line search for the step size for X, returns the new step size
@inline function _threadedproxStepX!(g::AbstractGLRM, params::ProxGradParams,
                            newX::Matrix{Float64}, gx::Matrix{Float64},
                            XY::Array{VecOrMat{Float64}}, newXY::Array{VecOrMat{Float64}},
                            αx::Number)
  #l = 1.5
  l = maximum(map(length, g.observed_features))+1#(mapreduce(length,+,g.observed_features) + 1)

  obj = sparse_loss_objective(g,XY) + evaluate(g.rx, g.X)
  newobj = NaN
  while αx > params.min_stepsize #Linesearch to find the new step size
    stepsize = αx/l

    axpy!(-stepsize, gx, newX)
    prox!(g.rx, newX, stepsize)
    reconstruct_obs!(g, newXY, X=newX)#At_mul_B!(newXY, newX, g.Y)
    newobj = sparse_loss_objective(g, newXY) + evaluate(g.rx, newX)
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
                              XY::Array{VecOrMat{Float64}}, newXY::Array{VecOrMat{Float64}},
                              αy::Number)
  #l = 1.5
  l = maximum(map(length, g.observed_examples)) + 1#(mapreduce(length,+,g.observed_features) + 1)
  #obj = threaded_objective(g,XY)
  obj = sparse_loss_objective(g, XY) + evaluate(g.ry, g.Y)
  newobj = NaN
  while αy > params.min_stepsize #Linesearch to find the new step size
    stepsize = αy/l
    axpy!(-stepsize, gy, newY)
    prox!(g.ry, newY, stepsize)
    reconstruct_obs!(g, newXY, Y=newY)#At_mul_B!(newXY, g.X, newY)
    newobj = sparse_loss_objective(g, newXY) + evaluate(g.ry, newY)
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
  XY = reconstruct_obs(g)

  tm = 0
  update_ch!(ch, tm, sparse_whole_objective(g,XY))
  (objx, objy) = NaN, NaN
  #Step sizes
  αx = params.stepsize
  αy = params.stepsize

  # stopping criterion: stop when decrease in objective < tol, scaled by the number of observations
  scaled_abs_tol = params.abs_tol * mapreduce(length,+,g.observed_features)

  newX = copy(X)
  newY = copy(Y)
  newXY = deepcopy(XY) #It's a ragged array of matrices, so deepcopy is needed

  #Gradient matrices
  gx = zeros(X)
  gy = zeros(Y)

  if verbose println("Fitting GGLRM") end
  for t in 1:params.max_iter
    #X update-----------------------------------------------------------------
    for xit in 1:params.inner_iter_X
      #_threadedupdateGradX!(g,XY,gx)
      _threadedupdateGradX!(g,XY,gx)
      #Take a prox step with line search
      αx, objx = _threadedproxStepX!(g, params, newX, gx, XY, newXY, αx)
      reconstruct_obs!(g, XY) #Get the new XY matrix for objective
    end
    #Y Update---------------------------------------------------------------
    for yit in 1:params.inner_iter_Y
      #_threadedupdateGradY!(g,XY,gy)
      _threadedupdateGradY!(g,XY,gy)
      αy, objy = _threadedproxStepY!(g, params, newY, gy, XY, newXY, αy)
      reconstruct_obs!(g, XY) #Get the new XY matrix for objective
    end
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

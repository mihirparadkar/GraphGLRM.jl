@inline function _updateGradX!(g::AbstractGLRM, XY::Matrix{Float64}, gx::Matrix{Float64})
  yidxs = get_yidxs(g.losses)
  #scale the gradient to zero
  scale!(gx,0)

  #Update the gradient
  for i in 1:size(g.X,2)
    gxi = view(gx, :, i)
    for j in g.observed_features[i]
      curgrad = grad(g.losses[j], XY[i,yidxs[j]], g.A[i,j])
      if isa(curgrad, Number)
        axpy!(curgrad, view(g.Y,:,yidxs[j]), gxi[:])
      else
        gemm!('N', 'N', 1.0, view(g.Y,:,yidxs[j]), curgrad, 1.0, gxi)
      end
    end
  end
end

@inline function _updateGradY!(g::AbstractGLRM, XY::Matrix{Float64}, gy::Matrix{Float64})
  yidxs = get_yidxs(g.losses)
  #scale the gradient to zero
  scale!(gy, 0)

  #Update the gradient
  for j in 1:length(g.losses)
    gyj = view(gy, :, yidxs[j])
    for i in g.observed_examples[j]
      curgrad = grad(g.losses[j], XY[i, yidxs[j]], g.A[i,j])
      if isa(curgrad, Number)
        axpy!(curgrad, view(g.X, :, i), gyj)
      else
        gemm!('N', 'T', 1.0, view(g.X, :, i), curgrad, 1.0, gyj)
      end
    end
  end
end


#Does a line search for the step size for X, returns the new step size
@inline function _wholeProxStepX!(g::AbstractGLRM, params::ProxGradParams,
                              newX::Matrix{Float64}, gx::Matrix{Float64},
                              XY::Matrix{Float64}, newXY::Matrix{Float64},
                              αx::Number)
    #l = 1.5
    l = maximum(map(length, g.observed_features))+1#(mapreduce(length,+,g.observed_features) + 1)
    #obj = threaded_objective(g,XY)
    obj = whole_objective(g,XY)
    newobj = 0.
    while αx > params.min_stepsize #Linesearch to find the new step size
      stepsize = αx/l

      axpy!(-stepsize, gx, newX)
      prox!(g.rx, newX, stepsize)
      At_mul_B!(newXY, newX, g.Y)
      newobj = whole_objective(g, newXY)
      #newobj = threaded_objective(g, newXY)
      if newobj < obj
        copy!(g.X, newX)
        αx *= (1.05)
        break
      else #Try again with smaller step-size
        copy!(newX, g.X)
        αx *= 0.7
        if αx < params.min_stepsize
          αx = params.min_stepsize * 1.1
          break
        end #if
      end #if else
    end #while
    αx, newobj
end

@inline function _rowProxStepX!(glrm::GGLRM, params::ProxGradParams,
                                newX::Matrix{Float64}, gx::Matrix{Float64},
                                obj_by_row::Array{Float64,1}, alpharow::Array{Float64,1})
  for i in 1:size(glrm.X,2)
    Xi = view(glrm.X, :, i)
    newXi = view(newX, :, i)
    gxi = view(gx, :, i)
    # take a proximal gradient step to update view(X,:,i)
    l = length(glrm.observed_features[i]) + 1 # if each loss function has lipshitz constant 1 this bounds the lipshitz constant of this example's objective
    obj_by_row[i] = row_objective(glrm, i, Xi) # previous row objective value
    while alpharow[i] > params.min_stepsize
      #Divide by lipshitz constant
      stepsize = alpharow[i]/l
      #Take gradient step and prox step
      axpy!(-stepsize, gxi, newXi)
      prox!(glrm.rx, newXi, stepsize)
      if row_objective(glrm, i, newXi) < obj_by_row[i]
        copy!(Xi, newXi)
        alpharow[i] *= 1.05 # choose a more aggressive stepsize
        break
      else # the stepsize was too big; undo and try again only smaller
        copy!(newXi, Xi)
        alpharow[i] *= .7 # choose a less aggressive stepsize
        if alpharow[i] < params.min_stepsize
            alpharow[i] = params.min_stepsize * 1.1
            break
        end
      end
    end
  end
end

@inline function _proxStepY!(glrm::GGLRM, params::ProxGradParams,
                              newY::Matrix{Float64}, gy::Matrix{Float64},
                              obj_by_col::Dict, alphacol::Dict)
  yidxs = get_yidxs(glrm.losses)
  for r in keys(glrm.ry)
    j = glrm.ry[r]
    #For a single-column regularizer
    if isa(j, Int)
      Yr = view(glrm.Y, :, yidxs[j])
      newYr = view(newY, :, yidxs[j])
      gyr = view(gy, :, yidxs[j])
      l = length(glrm.observed_examples[j]) + 1
    #For a multi-column regularizer
    else
      Yr = view(glrm.Y, :, j)
      newYr = view(newY, :, j)
      gyr = view(gy, :, j)
      l = maximum(map(length,glrm.observed_examples[j])) + 1
    end
    obj_by_col[r] = col_objective(glrm, r, newY)
    while alphacol[r] > params.min_stepsize
      stepsize = alphacol[r]/l
      axpy!(-stepsize, gyr, newYr)
      prox!(r, newYr, stepsize)
      if col_objective(glrm, r, newY) < obj_by_col[r]
        copy!(Yr, newYr)
        alphacol[r] *= 1.05 # choose a more aggressive stepsize
        break
      else # the stepsize was too big; undo and try again only smaller
        copy!(newYr, Yr)
        alphacol[r] *= .7 # choose a less aggressive stepsize
        if alphacol[r] < params.min_stepsize
            alphacol[r] = params.min_stepsize * 1.1
            break
        end
      end
    end
  end
end

### FITTING
function fit!(glrm::GGLRM, params::ProxGradParams;
			  ch::ConvergenceHistory=ConvergenceHistory("ProxGradGLRM"),
			  verbose=true,
			  kwargs...)
  ### initialization
  A = glrm.A # rename these for easier local access
  losses,rx,ry = glrm.losses, glrm.rx, glrm.ry
  X = glrm.X; Y = glrm.Y
  # check that we didn't initialize to zero (otherwise we will never move)
  if (vecnorm(Y) == 0) Y = .1*randn(k,d) end
  k = glrm.k
  m,n = size(A)

  # find spans of loss functions (for multidimensional losses)
  yidxs = get_yidxs(losses)
  d = maximum(yidxs[end])

  XY = At_mul_B(X,Y)
  # step size (will be scaled below to ensure it never exceeds 1/\|g\|_2 or so for any subproblem)
  alpharow = params.stepsize*ones(m)
  alphacol = Dict(r => params.stepsize for r in keys(ry))
  #For graph regularizer on X
  alphaX = 1
  # stopping criterion: stop when decrease in objective < tol, scaled by the number of observations
  scaled_abs_tol = params.abs_tol * mapreduce(length,+,glrm.observed_features)

  # alternating updates of X and Y
  if verbose println("Fitting GLRM") end
  update_ch!(ch, 0, whole_objective(glrm, XY))
  tm = time()
  steps_in_a_row = 0
  # gradient wrt columns of X
  gx = zeros(X)
  # gradient wrt column-chunks of Y
  gy = zeros(k, d)

  # working variables
  newX = copy(X)
  newY = copy(Y)

  # rowwise objective value
  obj_by_row = zeros(m)
  # columnwise objective value
  obj_by_col = Dict(r => 0. for r in keys(ry))
  # total objective value
  obj = 0.

  for t=1:params.max_iter
  ################################### X UPDATE ##############################
    #X gradient step
    _updateGradX!(glrm, XY, gx)
    #X prox step
    #Check if there is a matrix regularizer on X
    if (isa(glrm.rx, lastentry1) && isa(glrm.rx.r, MatrixRegularizer)) || isa(glrm.rx, MatrixRegularizer)
      alphaX, obj = _wholeProxStepX!(g, params, newX, gx, XY, newXY, alphaX)
    else
      _rowProxStepX!(glrm, params, newX, gx, obj_by_row, alpharow)
    end
    At_mul_B!(XY, X, Y) #Get the new XY matrix for objective
    obj = whole_objective(glrm, XY)
    println("Iteration $t, Updated X, objective value: $obj")
  ################################### Y UPDATE ##############################
    #Y gradient step
    _updateGradY!(glrm, XY, gy)
    #Y prox step
    _proxStepY!(glrm, params, newY, gy, obj_by_col, alphacol)
    At_mul_B!(XY, X, Y)
    obj = whole_objective(glrm, XY)
    println("Iteration $t, Updated Y, objective value: $obj")
    #=if t % 10 == 0
      println("Iteration $t, objective value: $obj")
    end=#
    #Update convergence history
    tm = time() - tm
    update_ch!(ch, tm, obj)
    tm = time()
    #Check stopping criterion
    obj_decrease = ch.objective[end-1] - obj
    if t>10 && (obj_decrease < scaled_abs_tol || obj_decrease/obj < params.rel_tol)
        break
    end
  end
end

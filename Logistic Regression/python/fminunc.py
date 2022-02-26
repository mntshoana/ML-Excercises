import numpy as np
import pandas as pd

def fminunc (func, X = None, options = None):
# X must be a matrix
   ## Get default options if requested.
  if (X is None and options == None and func == 'defaults'):
    # return default settings for options
    return { 
      "MaxIter": 400,
      "MaxFunEvals": float("inf"),
      "GradObj": False,
      "TolX": 1e-7,
      "TolFun": 1e-7,
      "OutputFunc": [], 
      "FunValCheck": False,
      "FinDiffType": "central",
      "TypicalX": [], 
      "AutoScaling": False
    }
    
  if isinstance(X, list):
    X = np.array(X)
  if not ( isinstance(X, (np.ndarray, np.generic)) ):
    print("Error: Invalid call to fminunc: X cannot be ", X)
    return

  if not ( isinstance(options, dict) or option != None ):
    print("Error: Invalid call to fminunc: options cannot be ", options)
    return 

  x_size = X.shape
  n = X.size

  has_grad = options.get("GradObj") == True
  cdif = options.get("GinDiffType", "central") == "central"
  max_iter = options.get("MaxIter", 400)
  max_fev = options.get("MaxFunEvals", float("inf"))
  out_func = options.get("OutputFunc")

  ## Get scaling matrix using the TypicalX option. If set to "auto", the
  ## scaling matrix is estimated using the jacobian.
  typical_x = options.get("TypicalX")
  if (typical_x == None):
    typical_x = np.ones((n, 1))
  
  autoscale = options.get("AutoScaling") == True
  dg = 1
  if (not autoscale):
    dg = 1 / typical_x
  
  funvalchk = options.get("FunValCheck") == True
  if (funvalchk):
    ## Replace func with a guarded version.
    func = lambda x: guarded_eval (func, x)

  macheps = 0
  if (np.isreal(X.dtype)):
    macheps = np.finfo(X.dtype).eps

  tolx = options.get("TolX", 1e-7)
  tolf = options.get("TolFun", 1e-7)

  factor = 0.1
  autodg = True

  niter = 1
  nfev = 0

  x = np.c_[X].copy()
  info = 0

  ## Initial evaluation.
  if has_grad:
    fval,_ = func (x)
  else:
    fval = func(x)
  n = x.shape[0]

  info = 0
  if out_func:
    optimvalues = {
      "iter": niter,
      "funccount": nfev,
      "fval": fval,
      "searchdirection": np.zeros((n, 1))
    }
    state = 'init'
    stop = out_func (x, optimvalues, state)
    if stop:
      info = -1

  nsuciter = 0
  lastratio = 0

  grad = np.array([])

  ## Outer loop.t
  while (niter < max_iter and nfev < max_fev and not info):

    grad0 = grad.copy()

    ## Calculate function value and gradient (possibly via FD).
    if has_grad:
      fval, grad = func (np.reshape(x, x_size))
      grad = grad.copy()
      nfev += 1
    else:
      grad = __fdjac__ (func, np.reshape(x, x_size), fval, typical_x, cdif).copy()
      nfev += (1 + cdif) * x.shape[0]

    if niter == 1:
      ## Initialize by identity matrix.
      hesr = np.eye (n)
    else:
      ## Use the damped BFGS formula.
      y = grad - grad0
      sBs = np.sum(x * np.conj(x))

      # NOTE: w is set later in the inner loop
      Bs = hesr.dot(w)
      sy = y.T.dot(s)
      theta = 0.8 / np.maximum(1 - np.divide(sy, sBs), 0.8)
      r = theta * y + (1-theta) *  Bs
      tempRoot = np.sqrt(s.T.dot(r))
      hesr = cholupdate( hesr, np.divide( r, tempRoot ), "+").T
      tempRoot = np.sqrt(sBs)
      hesr, info = cholupdate (hesr, np.divide(Bs, tempRoot ), "-")

      if info:
        hesr = np.eye(n)

    if autoscale:
      ## Second derivatives approximate the hessian.
      d2f = np.linalg.norm(hesr, ord=2, axis=0).T
      if niter == 1:
        dg = d2f
      else:
        dg = np.maximum(0.1*dg, d2f)

    if niter == 1:
      xn = np.linalg.norm(dg * x, ord=2)      
      delta = factor * np.maximum(xn, 1)

    if (np.linalg.norm(grad, ord=2) <= tolf * n*n).any():
      info = 1
      break

    suc = False
    decfac = 0.5

    ## Inner loop.
    while (not suc and niter <= max_iter and nfev < max_fev and not info):
      s = - __doglegm__ (hesr, grad, dg, delta)
      sn = np.linalg.norm (dg * s, ord=2)
      if niter == 1:
        delta = np.minimum(delta, sn)

      if has_grad:
        fval1, _ = func (np.reshape (x + s, x_size))
        fval1 = fval1.copy()
      else:
        fval1 = func (np.reshape (x + s, x_size)).copy()
      nfev += 1
      if (fval1 < fval):
        ## Scaled actual reduction.
        actred =  (fval - fval1) / (np.absolute(fval1) + np.absolute(fval))
      else:
        actred = -1
      
      
      w = hesr.dot(s)
      ## Scaled predicted reduction, and ratio.
      print("s")
      print(s)
      
      t = 1/2 * np.sum(w * np.conj(w)) + grad.T.dot(s)
      
      if (t < 0):
        prered = -t / ( np.absolute(fval) + np.absolute(fval + t))
        ratio = actred / prered
      else:
        prered = 0
        ratio = 0
    

      ## Update delta.
      if (ratio < np.minimum(np.maximum(0.1, 0.8*lastratio), 0.9)):
        delta =  np.dot(delta, decfac)
        decfac = np.power(decfac, 1.4142)
        if (delta <= 10 * macheps * xn):
          ## Trust region became uselessly small.
          info = -3
          break

      else:
        lastratio = ratio
        decfac = 0.5
        if (abs( 1-ratio ) <= 0.1):
          delta = 1.4142 * sn
        elif ratio >= 0.5:
          delta = np.maximum (delta, 1.4142 * sn)
 

      if (ratio >= 0.0001):
        ## Successful iteration.
        x += s
        xn = np.linalg.norm (dg * x, ord=2)
        fval = fval1
        nsuciter += 1
        suc = True


      niter += 1

      ## outputfcn only called after a successful iteration
      if out_func:
        optimvalues = {
          "iter": niter,
          "funccount": nfev,
          "fval": fval,
          "searchdirection": s
        }

        state = 'iter'
        stop = out_func (x, optimvalues, state)
        if stop:
          info = -1
          break
 

      ## Tests for termination conditions. A mysterious place, anything
      ## can happen if you change something here...

      ## The rule of thumb (which I'm not sure M*b is quite following)
      ## is that for a tolerance that depends on scaling, only 0 makes
      ## sense as a default value. But 0 usually means uselessly long
      ## iterations, so we need scaling-independent tolerances wherever
      ## possible.

      ## The following tests done only after successful step.
      if ratio >= 0.0001:
        ## This one is classic. Note that we use scaled variables again,
        ## but compare to scaled step, so nothing bad.
        if sn <= tolx * xn:
          info = 2
          ## Again a classic one.
        elif actred < tolf:
          info = 3
        
      #endif

    #end of while loop
  #end of while loop

  ## Restore original shapes.
  x = np.reshape(x, x_size)

  output = {
    "iterations": niter,
    "successful": nsuciter,
    "funcCount": nfev,
  }


  hess = np.dot(hesr.T, hesr)
  return x, fval, info, output, grad, hess




## An assistant function that evaluates a function handle and checks for
## bad results.
def guarded_eval (func, X = None):
  if (X):
    fx, gx = func(X)
  else:
    fx = func(X)
    gx = []

  if ( True in pd.DataFrame(fx).isna() ):
    print("Error: NaN value encountered in fminunc")
  return fx, gx




def __fdjac__ (func, x, fvec, typicalx, cdif, err = 0):
  eps = np.finfo(float).eps
  x = np.array(x)
  if cdif:
    err = np.power(max(eps, err), 1/3)
    h = np.dot(typicalx, err)
    fjac = np.zeros ((len(fvec), x.size))
    for i in range(0, x.size):
      x1 = x.copy()
      x2 = x.copy()
      x1[i] += h[i]
      x2[i] -= h[i]
      fjac[:, i] = np.divide(( np.c_[func(x1)] - np.c_[func(x2)]), (x1[i] - x2[i]))
    
  else:
    err = np.sqrt (max (eps, err))
    h = np.dot(typicalx, err)
    fjac = np.zeros ((len(fvec), x.size))
    for i in range(0, x.size):
      x1 = x.copy()
      x1[i] += h[i]
      fjac[:, i] = np.divide(( np.c_[func(x1)] - fvec), (x1[i] - x[i]))
  return fjac




## Solve the double dogleg trust-region minimization problem:
## Minimize 1/2*norm(r*x)^2  subject to the constraint norm(d.*x) <= delta,
## x being a convex combination of the gauss-newton and scaled gradient.

def __doglegm__ (r, grad, d, delta):
  ## Get Gauss-Newton direction.
  grad = grad.reshape(grad.size, 1)
  b = np.linalg.lstsq(r.T, grad)[0]
  x = np.linalg.lstsq(r,   b)[0]

  xn = np.linalg.norm(d * x, ord=2)

  if xn > delta:
    ## GN is too big, get scaled gradient.
    s = grad / d
    sn = np.linalg.norm (s, ord=2)
    if sn > 0:
      ## Normalize and rescale.
      s = np.divide(s, sn) / d
      ## Get the line minimizer in s direction.
      tn = np.linalg.norm ( np.dot(r,s), ord=2 )
      snm = np.divide(np.divide(sn, tn),  tn)
      if snm < delta:
        ## Get the dogleg path minimizer.
        bn = np.linalg.norm (b, ord=2)
        dxn = np.divide(delta, xn)
        snmd = np.divide(snm, delta)
        t = np.dot(np.dot(np.divide(bn,sn),  np.divide(bn,xn)),  snmd)
        t -= dxn * np.power(snmd,2) - np.sqrt( np.power(t-dxn, 2) + np.dot( np.power(1-dxn, 2), np.power(1-snmd, 2) ))
        alpha = np.divide( np.dot(dxn , 1- np.power(snmd, 2)), t)
      else:
        alpha = 0
    else:
      alpha = np.divide(delta, xn)
      snm = 0

    ## Form the appropriate convex combination.
    tempProduct = np.dot(1-alpha, np.minimum(snm, delta) )
    x = np.dot( alpha, x) + np.dot( tempProduct, s)  
    #end very first if
  return x


def cholupdate(R, x, sign):
  if not np.all(np.linalg.eigvals(R) > 0): #check that all eigenvalues are positive:
    return R, 1
  if (round(np.linalg.det(R)) == 0): # singular matrix
    return R, 2


  L = R.copy()
  x = x.copy()
  n = x.size
  for k in range(0, n):
    if sign == '+':
      r = np.sqrt(np.power(L[k, k], 2) + np.power(x[k],2))
    elif sign == '-':
      r = np.sqrt(np.power(L[k, k], 2) - np.power(x[k],2))
    c = np.divide(r, L[k, k])
    s = np.divide(x[k], L[k, k])
    L[k, k] = r
    if k < n:
      if sign == '+':
        result = np.c_[L[k+1:n, k]] + np.divide(s * x[k+1:n], c) 
      elif sign == '-':
        result = np.c_[L[k+1:n, k]] - np.divide(s * x[k+1:n], c)

      for l in range(0, result.size):
        L[k+1+l, k] = result[l]
      for l in range (k+1, n):
        x[l]= c * x[l] - s * L[l, k]
  if (sign == '-'):
    return L, 0
  return L

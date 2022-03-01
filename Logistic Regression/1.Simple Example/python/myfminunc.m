
function [x, fval, info, output, grad, hess] = myfminunc (X, y, x0, fcn = @(theta_)( costFunction(X, y, theta_)))

  # Create options structure for optimization function internally as python is unable to externally
  options = optimset('GradObj', 'on',  # seek fminunc to also return the first derivative of gradient decent as a second arg returned will be
                      'MaxIter', 400)


  ## Get default options if requested.
  if (nargin == 1 && ischar (fcn) && strcmp (fcn, 'defaults'))
    x = optimset ("MaxIter", 400, "MaxFunEvals", Inf, ...
    "GradObj", "off", "TolX", 1e-7, "TolFun", 1e-7,
    "OutputFcn", [], "FunValCheck", "off",
    "FinDiffType", "central",
    "TypicalX", [], "AutoScaling", "off");
    return;
  endif


  if (ischar (fcn))
    fcn = str2func (fcn, "global");
  endif

  xsiz = size (x0);
  n = numel (x0);



  has_grad = strcmpi (optimget (options, "GradObj", "off"), "on");
  cdif = strcmpi (optimget (options, "FinDiffType", "central"), "central");
  maxiter = optimget (options, "MaxIter", 400);
  maxfev = optimget (options, "MaxFunEvals", Inf);
  outfcn = optimget (options, "OutputFcn");


  ## Get scaling matrix using the TypicalX option. If set to "auto", the
  ## scaling matrix is estimated using the jacobian.
  typicalx = optimget (options, "TypicalX");
  if (isempty (typicalx))
    typicalx = ones (n, 1);
  endif

  autoscale = strcmpi (optimget (options, "AutoScaling", "off"), "on");
  if (! autoscale)
    dg = 1 ./ typicalx;
  endif

  funvalchk = strcmpi (optimget (options, "FunValCheck", "off"), "on");

  if (funvalchk)
    ## Replace fcn with a guarded version.
    fcn = @(x) guarded_eval (fcn, x);
  endif

  ## These defaults are rather stringent. I think that normally, user
  ## prefers accuracy to performance.

  macheps = eps (class (x0));

  tolx = optimget (options, "TolX", 1e-7);
  tolf = optimget (options, "TolFun", 1e-7);

  factor = 0.1;
  ## FIXME: TypicalX corresponds to user scaling (???)
  autodg = true;

  niter = 1;
  nfev = 0;


  x = x0(:);
  info = 0;

  ## Initial evaluation.
  fval = fcn (reshape (x, xsiz));
  n = length (x);

  if (! isempty (outfcn))
    optimvalues.iter = niter;
    optimvalues.funccount = nfev;
    optimvalues.fval = fval;
    optimvalues.searchdirection = zeros (n, 1);
    state = 'init';
    stop = outfcn (x, optimvalues, state);
    if (stop)
      info = -1;
    endif
  endif

  nsuciter = 0;
  lastratio = 0;

  grad = [];

  ## Outer loop.
  while (niter < maxiter && nfev < maxfev && ! info)

    grad0 = grad;

    ## Calculate function value and gradient (possibly via FD).
    if (has_grad)
      [fval, grad] = fcn (reshape (x, xsiz));
      grad = grad(:);
      nfev ++;
    else
      grad = __fdjac__ (fcn, reshape (x, xsiz), fval, typicalx, cdif)(:);
      nfev += (1 + cdif) * length (x);
    endif

    if (niter == 1)
      ## Initialize by identity matrix.
      hesr = eye (n);
    else
      ## Use the damped BFGS formula.
      y = grad - grad0;
      sBs = sumsq (w);
      Bs = hesr'*w;
      sy = y'*s;
      theta = 0.8 / max (1 - sy / sBs, 0.8);
      r = theta * y + (1-theta) * Bs;

      hesr = cholupdate (hesr, r / sqrt (s'*r), "+");
      disp ("hesr")
      disp (hesr)
      [hesr, info] = cholupdate (hesr, Bs / sqrt (sBs), "-");
      disp ("hesr")
      disp (hesr)
      if (info)
        hesr = eye (n);
      endif
    endif

    
    if (autoscale)
      ## Second derivatives approximate the hessian.
      print("autoscaling")
      d2f = norm (hesr, 'columns').';
      if (niter == 1)
        dg = d2f;
      else
        ## FIXME: maybe fixed lower and upper bounds?
        dg = max (0.1*dg, d2f);
      endif
    endif

    if (niter == 1)
      xn = norm (dg .* x);
      ## FIXME: something better?
      delta = factor * max (xn, 1);
    endif

    ## FIXME -- why tolf*n*xn? If abs (e) ~ abs(x) * eps is a vector
    ## of perturbations of x, then norm (hesr*e) <= eps*xn, i.e. by
    ## tolf ~ eps we demand as much accuracy as we can expect.
    if (norm (grad) <= tolf*n*xn)
      info = 1;
      break;
    endif

    suc = false;
    decfac = 0.5;

    ## Inner loop.
    while (! suc && niter <= maxiter && nfev < maxfev && ! info)
      
      s = - __doglegm__ (hesr, grad, dg, delta);

      sn = norm (dg .* s);
      if (niter == 1)
        delta = min (delta, sn);
      endif

      fval1 = fcn (reshape (x + s, xsiz)) (:);
      nfev ++;
      
      if (fval1 < fval)
        ## Scaled actual reduction.
        actred =  (fval - fval1) / (abs (fval1) + abs (fval));
      else
        actred = -1;
      endif
      
      w = hesr*s;
      ## Scaled predicted reduction, and ratio.
      
      t = 1/2 * sumsq (w) + grad'*s;
      if (t < 0)
        prered = -t/(abs (fval) + abs (fval + t));
        ratio = actred / prered;
      else
        prered = 0;
        ratio = 0;
      endif

      ## Update delta.
      if (ratio < min(max(0.1, 0.8*lastratio), 0.9))
        delta *= decfac;
        decfac ^= 1.4142;
        if (delta <= 1e1*macheps*xn)
          ## Trust region became uselessly small.
          info = -3;
          break;
        endif
      else
        lastratio = ratio;
        decfac = 0.5;
        if (abs (1-ratio) <= 0.1)
          delta = 1.4142*sn;
        elseif (ratio >= 0.5)
          delta = max (delta, 1.4142*sn);
        endif
      endif

      if (ratio >= 1e-4)
        ## Successful iteration.
        x += s;
        xn = norm (dg .* x);
        fval = fval1;
        nsuciter ++;
        suc = true;
      endif

      niter ++;

      ## FIXME: should outputfcn be only called after a successful iteration?
      if (! isempty (outfcn))
        optimvalues.iter = niter;
        optimvalues.funccount = nfev;
        optimvalues.fval = fval;
        optimvalues.searchdirection = s;
        state = 'iter';
        stop = outfcn (x, optimvalues, state);
        if (stop)
          info = -1;
          break;
        endif
      endif

      ## Tests for termination conditions. A mysterious place, anything
      ## can happen if you change something here...

      ## The rule of thumb (which I'm not sure M*b is quite following)
      ## is that for a tolerance that depends on scaling, only 0 makes
      ## sense as a default value. But 0 usually means uselessly long
      ## iterations, so we need scaling-independent tolerances wherever
      ## possible.

      ## The following tests done only after successful step.
      if (ratio >= 1e-4)
        ## This one is classic. Note that we use scaled variables again,
        ## but compare to scaled step, so nothing bad.
        if (sn <= tolx*xn)
          info = 2;
          ## Again a classic one.
        elseif (actred < tolf)
          info = 3;
        endif
      endif

    endwhile
  endwhile

  ## Restore original shapes.
  x = reshape (x, xsiz);

  output.iterations = niter;
  output.successful = nsuciter;
  output.funcCount = nfev;

  if (nargout > 5)
    hess = hesr'*hesr;
  endif

endfunction

## An assistant function that evaluates a function handle and checks for
## bad results.
function [fx, gx] = guarded_eval (fun, x)
  if (nargout > 1)
    [fx, gx] = fun (x);
  else
    fx = fun (x);
    gx = [];
  endif

  if (! (isreal (fx) && isreal (gx)))
    error ("fminunc:notreal", "fminunc: non-real value encountered");
  elseif (any (isnan (fx(:))))
    error ("fminunc:isnan", "fminunc: NaN value encountered");
  endif
endfunction

%!function f = __rosenb (x)
%!  n = length (x);
%!  f = sumsq (1 - x(1:n-1)) + 100 * sumsq (x(2:n) - x(1:n-1).^2);
%!endfunction
%!test
%! [x, fval, info, out] = fminunc (@__rosenb, [5, -5]);
%! tol = 2e-5;
%! assert (info > 0);
%! assert (x, ones (1, 2), tol);
%! assert (fval, 0, tol);
%!test
%! [x, fval, info, out] = fminunc (@__rosenb, zeros (1, 4));
%! tol = 2e-5;
%! assert (info > 0);
%! assert (x, ones (1, 4), tol);
%! assert (fval, 0, tol);

## Solve the double dogleg trust-region minimization problem:
## Minimize 1/2*norm(r*x)^2  subject to the constraint norm(d.*x) <= delta,
## x being a convex combination of the gauss-newton and scaled gradient.

## TODO: error checks
## TODO: handle singularity, or leave it up to mldivide?

function x = __doglegm__ (r, g, d, delta)
  ## Get Gauss-Newton direction.
  b = r' \ g;
  x = r \ b;
  xn = norm (d .* x);
  
  if (xn > delta)
    ## GN is too big, get scaled gradient.
    s = g ./ d;
    sn = norm (s);
    if (sn > 0)
      ## Normalize and rescale.
      s = (s / sn) ./ d;
      ## Get the line minimizer in s direction.
      tn = norm (r*s);
      snm = (sn / tn) / tn;
      if (snm < delta)
        ## Get the dogleg path minimizer.
        bn = norm (b);
        dxn = delta/xn; snmd = snm/delta;
        t = (bn/sn) * (bn/xn) * snmd;
        t -= dxn * snmd^2 - sqrt ((t-dxn)^2 + (1-dxn^2)*(1-snmd^2));
        alpha = dxn*(1-snmd^2) / t;
      else
        alpha = 0;
      endif
    else
      alpha = delta / xn;
      snm = 0;
    endif
    ## Form the appropriate convex combination.
    x = alpha * x + ((1-alpha) * min (snm, delta)) * s;
  endif
endfunction


function fjac = __fdjac__ (fcn, x, fvec, typicalx, cdif, err = 0)
  if (cdif)
    err = (max (eps, err)) ^ (1/3);
    h = typicalx*err;
    fjac = zeros (length (fvec), numel (x));
    for i = 1:numel (x)
      x1 = x2 = x;
      x1(i) += h(i);
      x2(i) -= h(i);
      fjac(i) = (fcn (x1)(:) - fcn (x2)(:)) / (x1(i) - x2(i));
    endfor
  else
    err = sqrt (max (eps, err));
    h = typicalx*err;
    fjac = zeros (length (fvec), numel (x));
    for i = 1:numel (x)
      x1 = x;
      x1(i) += h(i);
      fjac(:,i) = (fcn (x1)(:) - fvec) / (x1(i) - x(i));
    endfor
  endif
endfunction

function [J, deltaJ] = costFunction(X, y, theta)
%   Logistic regression
%   Cost and gradient 
    m = length(y);
    
    % HYPOTHESIS
    z = X * theta;
    %FILE: sigmoid.m
    h = sigmoid(z);

    % COST
    a = - y' * log(h);
    b = (1 - y)' * log(1 - h);
    J = 1/m * (a - b);

    % GRADIENT
    deltaJ = 1/m * X' * ( h - y );

end

function sig = sigmoid(z)
    sig = 1 ./ (1 + exp(-z));
end

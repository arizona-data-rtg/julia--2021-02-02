######################################################
## LittleDiff.jl

# This file implements LittleDiff.jl, a Julia program that
# illustrates the basic ideas of forward-mode automatic
# differentiation.

# Copyright (C) 2021 by Kevin K Lin <klin@math.arizona.edu>

# This program is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation;
# either version 2 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public
# License along with this program; if not, write to the Free
# Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301 USA.

######################################################
## This file illustrates one way to implement (forward-mode)
## automatic differentiation.  It implements basically what
## we teach students in Calc I.

## The approach here is a variant on "dual numbers": a
## "Dual" is a pair (f,Df), where f is a given function and
## Df its derivative.  Mathematically, these can be viewed
## as the path of a particle along with its velocity, if we
## view f(t) as its position at time t.  This is different
## from the usual dual number, but more "functional" and (I
## think) more natural mathematically.  It also means the
## derivative function is fully formed at the time diff() is
## called, rather than on-the-fly when a dual number is
## passed through.

## Notes:

## 1) Methods.  This file doesn't actually define many new
## functions.  It mostly adds new "methods" for working with
## the new Dual type.  (In Julia, you can have multiple
## functions with the same name.  It figures out which one
## to use based on the type(s) of the arguments.)  Since
## operations like +, *, etc., are already defined in
## Julia's "Base" namespace, Julia requires that we "import"
## them first before extending them.  (This restriction
## exists for safety reasons.)

## 2) Symbolic representations.  To keep the code simple,
## I've left out many obvious optimizations.  The one
## "frill" is that I added methods for applying functions to
## "symbols", which are basically short strings you can type
## with `:foo'.  (Symbols are a little easier to type and
## have properties that make them better suited for symbolic
## expressions.)  So that you can do something like
## `sin(cos(:x))' and get back a symbolic representation of
## sin(cos(x)).  This is useful for debugging the
## differentiation code, say for checking
## `diff(x->sin(cos(x)))(:x)' is right.

## 3) For those who know some differential geometry and AD:
## this category of objects are naturally pushed forward,
## via forward-mode automatic differentiation.  One can try
## to implement forms (maybe this is closer to backward
## AD?), but that doesn't quite work because the functions
## are composed in the wrong order.

## 4) Multivariate calculus.  This only implements
## single-variable functions and their derivatives.
## Multivariate functions and partial derivatives require a
## little more work (but can build on this framework).
## Also, various obvious optimizations are omitted to keep
## the code simple.


## Examples:
#=
julia> include("LittleDiff.jl")
newton (generic function with 1 method)

julia> diff(x->sin(cos(x)))
#2 (generic function with 1 method)

julia> diff(x->sin(cos(x)))(:x)
:(cos(cos(x)) * (-(sin(x)) * 1))

julia> diff(x->sin(cos(x)))(1)
-0.7216061490634433

julia> newton(sin,3)
(sol = 3.141592653589793, resid = 0.0, iter = 4)

=#

## Important: diff operates on functions, not symbolic
## expressions or names.  So `diff(sin(cos(1)))' will result
## in an error (because `sin(cos(1))' is just a number), as
## will `diff(sin(cos(:x)))' (because `sin(cos(:x))' is a
## symbolic expression).  The anonymous function
## `x->sin(cos(x))', on the other hand, does create the
## composition of sin and cos.

################################
## general stuff

import Base:diff,|

struct Dual
    basept::Function
    vector::Function
end

## how to evaluate univariate functions
function evaluate(f::Function, df::Function, w::Dual)
    Dual(x->f(w.basept(x)), x->df(w.basept(x))*w.vector(x))
end

## how to evaluate binary operations
function evaluate(f::Function, fx::Function, fy::Function, v::Dual, w::Dual)
    Dual(x->f(v.basept(x),w.basept(x)),
          x->(fx(v.basept(x),w.basept(x))*v.vector(x) +
              fy(v.basept(x),w.basept(x))*w.vector(x)))
end

neo               = Dual(identity,x->1)
ev(f::Function)   = f(neo).basept
diff(f::Function) = f(neo).vector

compose(f,g) = x->f(g(x))
|(f::Function, g::Function) = compose(g,f)

################################
## (mostly) binary operators
import Base:*,+,-,/,^

Literal = Union{Number,Symbol,Expr}

## operator +
+(v::Dual,w::Dual)       = evaluate(+,(x,y)->y,(x,y)->x,v,w)
+(a::Number,w::Dual)     = evaluate(x->a+x, x->1, w)
+(w::Dual,a::Number)     = evaluate(x->x+a, x->1, w)
+(X::Literal,Y::Literal) = :($X + $Y)

## operator -
-(v::Dual)               = evaluate(-,x->-1,v)
-(X::Literal)            = :(-$X)
-(v::Dual,w::Dual)       = evaluate(-, (x,y)->1, (x,y)->-1, v, w)
-(a::Number,w::Dual)     = evaluate(x->a-x, x->-1, w)
-(v::Dual,a::Number)     = evaluate(x->x-a, x->1, v)
-(X::Literal,Y::Literal) = :($X - $Y)

## operator *
*(v::Dual,w::Dual)       = evaluate(*, (x,y)->y, (x,y)->x, v, w)
*(v::Dual,w::Number)     = evaluate(x->w*x, x->w, v)
*(a::Number,w::Dual)     = evaluate(x->a*x, x->a, w)
*(X::Literal,Y::Literal) = :(($X)*($Y))

## operator /
/(v::Dual,w::Dual)       = evaluate(/, (x,y)->1/y, (x,y)->-x/y^2, v, w)
/(v::Dual,a::Number)     = evaluate(x->x/a, x->1/a, v)
/(a::Number,w::Dual)     = evaluate(x->a/x, x->-a/x^2, w)
/(X::Literal,Y::Literal) = :(($X)/($Y))

################################
## Functions

import Base:exp,log,cos,sin,tan,atan

## operator ^
function ^(v::Dual,a::Number)
    if a == 0
        Dual(x->0, x->0)
    else
        evaluate(x->x^a, x->a*x^(a-1), v)
    end
end
^(X::Literal,Y::Literal) = :(($X)^($Y))

## exp()
exp(w::Dual)    = evaluate(exp,exp,w)
exp(X::Literal) = :(exp($X))

## log()
log(w::Dual)    = evaluate(log, x->1/x, w)
log(X::Literal) = :(log($X))

## cos()
cos(w::Dual)    = evaluate(cos, x->-sin(x), w)
cos(X::Literal) = :(cos($X))

## sin()
sin(w::Dual)    = evaluate(sin, cos, w)
sin(X::Literal) = :(sin($X))

## tan()
tan(w::Dual)    = evaluate(tan, x->1/(cos(x)^2), w)
tan(X::Literal) = :(tan($X))

## atan()
atan(w::Dual)    = evaluate(atan, x->1/(1+x^2), w)
atan(X::Literal) = :(atan($X))

######################################################
## A more serious example

function newton(f,x;nmax=100,tol=eps())
    df = diff(f)
    resid = Inf
    for i=1:nmax
        xnew = x - f(x)/df(x)
        resid = abs(xnew-x)
        if  resid <= tol
            return (sol=xnew,resid=resid,iter=i)
        end
        x = xnew
    end
    return (sol=x,resid=resid,iter=nmax)
end

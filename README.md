# Quats

[![Build Status](https://travis-ci.org/spacedome/Quats.jl.svg?branch=master)](https://travis-ci.org/spacedome/Quats.jl)

[![Coverage Status](https://coveralls.io/repos/spacedome/Quats.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/spacedome/Quats.jl?branch=master)

[![codecov.io](http://codecov.io/github/spacedome/Quats.jl/coverage.svg?branch=master)](http://codecov.io/github/spacedome/Quats.jl?branch=master)

A Quaternion type for Julia.

This package was made with the goal of having a quaternion type for numerical analysis and linear algebra with quaternion matrices.
It is designed to match the Base library complex.jl as closely as possible (much of this code is taken from there and modified) and some care was taken to make it interoperable with the Base complex numbers.
An alternative package is JuliaGeometry/Quaternions.jl, which seems more focused on using them for representations of rotations and kinematics.
The differences are not huge, both are relatively small packages, though each has made incompatible syntactical choices, do not import both at once.


## Documentation

Complex.jl defines the complex unit `im` and similarly Quats.jl extends this with the additional complex units `jm` and `km`.
The quaternions are defined by the following algebraic properties.
```julia
im^2 == jm^2 == km^2 == im * jm * km == -1
```

Quaternions can be constructed using the units:
```julia
q = 1 + 2im + 6jm - km
q = 0.0 - 2.5jm + km + 10im
 ```
They can be made with the `Quaternion` constructor, which accepts either a `Real`, four `Real` numbers, a `Complex` number, two `Complex` numbers, or a length four `Real` vector. The function `quat` is a shorthand for the constructor, if you are willing to infer the type. The type is parameterized as `Quaternion{T <: Real} <: Number`.
```julia
q = Quaternion(1.0, 2.0, 3.0, 4.0)     # 1.0 + 2.0im + 3.0jm + 4.0km
q = Quaternion{Float32}(1.0)           # 1.0 + 0.0im + 0.0jm + 0.0km
q = Quaternion(1.0 + 2.0im)            # 1.0 + 2.0im + 0.0jm + 0.0km
q = Quaternion(1.0+2.0im, 3.0+4.0im)   # 1.0 + 2.0im + 3.0jm + 4.0km
q = quat(1.0)                          # 1.0 + 0.0im + 0.0jm + 0.0km
q = quat([1, 2, 3, 4])                 # 1 + 2im + 3jm + 4km
### make a matrix of quaternions
q = quat.([1, 2])                      # [1+0im+0jm+0km, 2+0im+0jm+0km]
```

The standard operators are implemented: `+, *, -, /`.  
Note that quaternion multiplication is non-commutative, meaning `q*w ≠ w*q` in general.

the functions `real` and `imag` work as they do for complex, and are extended by `jmag` and `kmag`.
These can also be accessed as `q.re`, `q.im`, `q.jm`, and `q.km`, respectively.
The function `complex` returns the the real and imag part as a complex number.
```julia
q = quat(1,2,3,4)
real(q) == q.re == 1
imag(q) == q.im == 2
jmag(q) == q.jm == 3
kmag(q) == q.km == 4
complex(1+2im+3jm+4km) == 1+2im == complex(1,2)
```

The conjugate, `conj` or `'`, for quaternions is defined similarly to complex numbers.
So is the inverse `inv`, as well as `abs` and `abs2`.
The absolute value, or norm, is taken to be the euclidean norm of the components, and `abs2` is just the absolute value squared.
Note that it can also be defined by `abs2(q) == q*conj(q) == conj(q)*q`.
If you need a different norm `p` you can always use `norm(vec(q), p)`.
The inverse is defined as `inv(q) == conj(q)/abs(q)`.
The function `normalize` returns the unit quaternion `q/abs(q)`.
```julia
conj(1 + 2im + 3jm + 4km) == 1 - 2jm - 3jm - 4km
abs(quat(2, 4, 4, 8)) == 10.0
abs(im+jm+km) == sqrt(3)
abs2(quat(2, 4, 4, 8)) == 100.0
inv(quat(1,1,1,1)) == 0.25 - 0.25im - 0.25jm - 0.25km
normalize(quat(2,2,2,2)) == 0.5 + 0.5im + 0.5jm + 0.5km
```



Using `rand` and `randn` one can generate random quaternions or arrays of random matrices.
With `rand` each component of the quaternion is uniformly sampled from [0,1) for floats (by default).
With `randn` each component is sampled independently from a standard normal.
They should be scaled to make the whole quaternion sampled from 4D normal (TODO).


The typical Boolean functions are defined `isreal, iszero, isinteger, isfinite, isnan, isinf, iszero, iscomplex, isequal, ==` and should behave as expected.
The `isapprox` or `≈` uses the generic definition which only depends on the absolute value (euclidean distance) of the two quaternions being compared `abs(q-w)`.

The following type and conversion functions are implemented `zero, float, big, widen, round, vec`.
The `vec` function just returns a vector of four real numbers, the components of the quaternion it is called on.
As for complex numbers, you can optionally specify a second rounding mode for the imaginary components, by default they are the same as for the real component.

The fundamental representations of quaternions are the complex representation `q = x + yj` where `x` and `y` are complex, and the matrix representation.
it is possible to represent a quaternion as a complex or real matrix, though the complex matrix is usually preferable.
(Similarly a complex number has a real matrix representation).
For a quaternion `q = x + yj` where `x,y` complex we can represent it as the complex matrix:
```julia
Q = [ x  y ;
     -y' x']
```
This representation preserves the algebraic structure of the quaternions (it is a homomorphism).
This representation can be computed with `cmatrix`, above we would have `Q == cmatrix(q)`.
We can do the same for a quaternion matrix to a complex matrix `Q = X + Yj`.
```julia
Q = [       X        Y ;
     -conj.(X) conj.(Y)]
```
This is also a homomorphism and preserves the algebraic structure of the quaternion matrices, as well as in a very useful sense preserving the most important spectral properties.
To convert back to quaternion from a complex matrix use `qmatrix`.


If you need `svd`, `svdvals`, or `cond` for condition numbers of quaternion matrices, GenericSVD.jl has been tested and works great.

The functions `exp` and `log` can be defined for quaternions.
See Neil Dantam - [Quaternion Computation](http://www.neil.dantam.name/note/dantam-quaternion.pdf) for the approach to implementation used here.
It is simple to then define `^` based on these, and subsequently `sqrt` as a special case.
There seems to be some small issues, such as `sqrt(-1)` evaluating to `1` which seems quite wrong.
For the most part `exp` and `log` seem to work, but I would not trust the values too much, in particular there may be some issues of numerical instability.
It is also possible to define trigonometric and other geometric functions, which are not currently implemented. See JuliaGeometry/Quaternions.jl if you need these (or let me know in the relevant issue, or put in a pull request).

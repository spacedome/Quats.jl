__precompile__()

module Quats

import Random.AbstractRNG
import Base: convert, promote_rule, show
import Base: real, complex, float, imag, isinteger, isfinite, isnan, isinf, iszero, isequal, zero
import Base: vec, conj, abs, abs2, inv, big, widen, rand, randn, exp, log, round, sqrt
import Base: +, -, *, /, ^, ==

export Quaternion
export QuaternionF16
export QuaternionF32
export QuaternionF64
export QuatF16
export QuatF32
export QuatF64

export quat
export jm
export km
export jmag
export kmag
export cmatrix
export qmatrix


include("quaternion.jl")

end # module

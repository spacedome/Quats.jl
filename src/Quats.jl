__precompile__()

module Quats

import Random: AbstractRNG, SamplerType
import LinearAlgebra: norm, normalize
import Base: convert, promote_rule, show, write, read
import Base: real, imag, complex
import Base: float, big, widen, bswap
import Base: isinteger, isfinite, isnan, isinf, iszero, isequal, isone, zero, in
import Base: vec, conj, abs, abs2, inv, rand, randn, exp, log, round, sqrt
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
export norm
export normalize


include("quaternion.jl")

end # module

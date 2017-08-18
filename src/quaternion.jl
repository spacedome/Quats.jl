importall Base

export Quaternion, quat, jm, km, Quaternion128, Quaternion256, Quat128, Quat256, jmag, kmag, cmatrix, qmatrix

"""
    Quaternion{T<:Real} <: Number
Quaternion number type with real and imaginary parts of type `T`.
`Quaternion256` and `Quaternion128` are aliases for
`Quaternion{Float64}` and `Quaternion{Float32}` respectively.
"""
struct Quaternion{T<:Real} <: Number
    re::T
    im::T
    jm::T
    km::T
end

Quaternion(a::Real, b::Real, c::Real, d::Real) = Quaternion(promote(a,b,c,d)...)
Quaternion(z::Complex) = Quaternion(real(z), imag(z), zero(real(z)), zero(real(z)) )
Quaternion(x::Real) = Quaternion(x, zero(x), zero(x), zero(x))
Quaternion(v::Vector{<:Real}) = Quaternion(v[1], v[2], v[3], v[4])

"""
    jm
Quaternion imaginary unit.
"""
const jm = Quaternion(false,false,true,false)
"""
    km
Quaternion imaginary unit.
"""
const km = Quaternion(false,false,false,true)

const Quaternion128 = Quaternion{Float32}
const Quaternion256 = Quaternion{Float64}
const Quat128 = Quaternion{Float32}
const Quat256 = Quaternion{Float64}

convert(::Type{Quaternion{T}}, x::Real) where {T<:Real} = Quaternion{T}(x,0,0,0)
convert(::Type{Quaternion{T}}, z::Complex) where {T<:Real} = Quaternion{T}(real(z),imag(z),0,0)

convert(::Type{Quaternion}, q::Quaternion) = q
convert(::Type{Quaternion}, z::Complex) = Quaternion(z)
convert(::Type{Quaternion}, x::Real) = Quaternion(x)


promote_rule(::Type{Quaternion{T}}, ::Type{S}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}
promote_rule(::Type{Quaternion{T}}, ::Type{Complex{S}}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}
promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{S}}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}


widen(::Type{Quaternion{T}}) where {T} = Quaternion{widen(T)}


real(q::Quaternion) = q.re
imag(q::Quaternion) = q.im

"""
    jmag(q)
Return the j-imaginary part of the quaternion number `q`.
```jldoctest
julia> jmag(1 + 3im + 2jm + 5km)
2
```
"""
jmag(q::Quaternion) = q.jm

"""
    kmag(q)
Return the k-imaginary part of the quaternion number `q`.
```jldoctest
julia> kmag(1 + 3im + 2jm + 5km)
5
```
"""
kmag(q::Quaternion) = q.km

real(::Type{Quaternion{T}}) where {T<:Real} = T

isreal(q::Quaternion) = iszero(q.im) && iszero(q.jm) && iszero(q.km)
isinteger(q::Quaternion) = isreal(q) && isinteger(q.re)
isfinite(q::Quaternion) = isfinite(q.re) && isfinite(q.im) && isfinite(q.jm) && isfinite(q.km)
isnan(q::Quaternion) = isnan(q.re) | isnan(q.im) | isnan(q.jm) | isnan(q.km)
isinf(q::Quaternion) = isinf(q.re) | isinf(q.im) | isinf(q.jm) | isinf(q.km)
iszero(q::Quaternion) = iszero(q.re) && iszero(q.im) && iszero(q.jm) && iszero(q.km)
iscomplex(q::Quaternion) = iszero(q.jm) && iszero(q.km)

"""
    quat(r, [i], [j], [k])
Convert real numbers or arrays to quaternion. `i`, `j`, `k`, default to zero.
"""
quat(q::Quaternion) = q
quat(x::Real) = Quaternion(x)
quat(z::Complex) = Quaternion(z)
quat(a::Real, b::Real, c::Real, d::Real) = Quaternion(a, b, c, d)
quat(v::Vector{<:Real}) = Quaternion(v)

complex(::Type{Quaternion{T}}) where {T<:Real} = Complex{T}
complex(q::Quaternion) = complex(q.re, q.im)


"""
    quat(T::Type)
Returns an appropriate type which can represent a value of type `T` as a quaternion.
Equivalent to `typeof(quat(zero(T)))`.
```jldoctest
julia> quat(Quaternion{Int})
Quaternion{Int64}
julia> quat(Int)
Quaternion{Int64}
```
"""
quat(::Type{T}) where {T<:Real} = Quaternion{T}
quat(::Type{Complex{T}}) where {T<:Real} = Quaternion{T}
quat(::Type{Quaternion{T}}) where {T<:Real} = Quaternion{T}

vec(q::Quaternion) = vcat(q.re, q.im, q.jm, q.km) 

function show(io::IO, q::Quaternion)
    r, i, j, k = vec(q)
    compact = get(io, :compact, false)
    show(io, r)
    if signbit(i) && !isnan(i)
        i = -i
        print(io, compact ? "-" : " - ")
    else
        print(io, compact ? "+" : " + ")
    end
    show(io, i)
    if !(isa(i,Integer) && !isa(i,Bool) || isa(i,AbstractFloat) && isfinite(i))
        print(io, "*")
    end
    print(io, "im")
    if signbit(j) && !isnan(j)
        j = -j
        print(io, compact ? "-" : " - ")
    else
        print(io, compact ? "+" : " + ")
    end
    show(io, j)
    if !(isa(j,Integer) && !isa(j,Bool) || isa(j,AbstractFloat) && isfinite(j))
        print(io, "*")
    end
    print(io, "jm")
    if signbit(k) && !isnan(k)
        k = -k
        print(io, compact ? "-" : " - ")
    else
        print(io, compact ? "+" : " + ")
    end
    show(io, k)
    if !(isa(k,Integer) && !isa(k,Bool) || isa(k,AbstractFloat) && isfinite(k))
        print(io, "*")
    end
    print(io, "km")
end

function show(io::IO, q::Quaternion{Bool})
    if q == im 
        print(io, "im")
    elseif q == jm 
        print(io, "jm")
    elseif q == km
        print(io, "km")
    else 
        print(io, "Quaternion($(q.re),$(q.im),$(q.jm),$(q.km))")
    end
end

function read(s::IO, ::Type{Quaternion{T}}) where T<:Real
    r = read(s,T)
    i = read(s,T)
    j = read(s,T)
    k = read(s,T)
    Quaternion{T}(r,i,j,k)
end
function write(s::IO, q::Quaternion)
    write(s,q.re,q.im,q.jm,q.km)
end

==(q::Quaternion, w::Quaternion) = q.re == w.re && q.im == w.im && q.jm == w.jm && q.km == w.km
==(q::Quaternion, z::Complex) = iscomplex(q) && q.im == imag(z) && q.re == real(z)
==(q::Quaternion, x::Real) = isreal(q) && q.re == x

isequal(q::Quaternion, w::Quaternion) = isequal(q.re,w.re) & isequal(q.im,w.im) &
                                        isequal(q.jm,w.jm) & isequal(q.km,w.km)

#TODO: hash

conj(q::Quaternion) = Quaternion(q.re,-q.im,-q.jm,-q.km)
abs(q::Quaternion)  = vecnorm([q.re q.im q.jm q.km])
abs2(q::Quaternion) = q.re*q.re + q.im*q.im + q.jm*q.jm + q.km*q.km 
inv(q::Quaternion)  = conj(q)/abs2(q)
inv(q::Quaternion{<:Integer}) = inv(float(q))

-(q::Quaternion) = Quaternion(-q.re, -q.im, -q.jm, -q.km)
+(q::Quaternion, w::Quaternion) = Quaternion(q.re + w.re, q.im + w.im,
                                             q.jm + w.jm, q.km + w.km)
-(q::Quaternion, w::Quaternion) = Quaternion(q.re - w.re, q.im - w.im,
                                             q.jm - w.jm, q.km - w.km)
*(q::Quaternion, w::Quaternion) = Quaternion(q.re*w.re - q.im*w.im - q.jm*w.jm - q.km*w.km,
                                             q.re*w.im + q.im*w.re + q.jm*w.km - q.km*w.jm,
                                             q.re*w.jm - q.im*w.km + q.jm*w.re + q.km*w.im,
                                             q.re*w.km + q.im*w.jm - q.jm*w.im + q.km*w.re)


# Why all this Bool code? (copied from complex.jl)
+(x::Bool, q::Quaternion{Bool}) = Quaternion(x + q.re, q.im, q.jm, q.km)
+(q::Quaternion{Bool}, x::Bool) = Quaternion(q.re + x, q.im, q.jm, q.km)
-(x::Bool, q::Quaternion{Bool}) = Quaternion(x - q.re, - q.im, - q.jm, - q.km)
-(q::Quaternion{Bool}, x::Bool) = Quaternion(q.re - x, q.im, q.jm, q.km)
*(x::Bool, q::Quaternion{Bool}) = Quaternion(x * q.re, x * q.im, x * q.jm, x * q.km)
*(q::Quaternion{Bool}, x::Bool) = Quaternion(q.re * x, q.im * x, q.jm * x, q.km * x)

+(z::Complex{Bool}, q::Quaternion{Bool}) = Quaternion(z.re + q.re, z.im + q.im, q.jm, q.km)
+(q::Quaternion{Bool}, z::Complex{Bool}) = Quaternion(q.re + z.re, q.im + z.im, q.jm, q.km)
-(z::Complex{Bool}, q::Quaternion{Bool}) = Quaternion(z.re - q.re, z.im - q.im, - q.jm, - q.km)
-(q::Quaternion{Bool}, z::Complex{Bool}) = Quaternion(q.re - z.re, q.im - z.im, q.jm, q.km)
*(q::Quaternion{Bool}, z::Complex{Bool}) = Quaternion(q.re*z.re - q.im*z.im, q.re*z.im + q.im*z.re,
                                                      q.jm*z.re + q.km*z.im, q.km*z.re - q.jm*z.im)
*(z::Complex{Bool}, q::Quaternion{Bool}) = Quaternion(z.re*q.re - z.im*q.im, z.re*q.im + z.im*q.re,
                                                      z.re*q.jm - z.im*q.km, z.re*q.km + z.im*q.jm)

+(x::Bool, q::Quaternion) = Quaternion(x + q.re, q.im, q.jm, q.km)
+(q::Quaternion, x::Bool) = Quaternion(q.re + x, q.im, q.jm, q.km)
-(x::Bool, q::Quaternion) = Quaternion(x - q.re, - q.im, - q.jm, - q.km)
-(q::Quaternion, x::Bool) = Quaternion(q.re - x, q.im, q.jm, q.km)
*(x::Bool, q::Quaternion) = Quaternion(x * q.re, x * q.im, x * q.jm, x * q.km)
*(q::Quaternion, x::Bool) = Quaternion(q.re * x, q.im * x, q.jm * x, q.km * x)

+(x::Real, q::Quaternion{Bool}) = Quaternion(x + q.re, q.im, q.jm, q.km)
+(q::Quaternion{Bool}, x::Real) = Quaternion(q.re + x, q.im, q.jm, q.km)
function -(x::Real, q::Quaternion{Bool})
    # we don't want the default type for -(Bool)
    re = x-q.re
    Quaternion(re, - oftype(re, q.im), - oftype(re, q.jm), - oftype(re, q.km))
end
-(q::Quaternion{Bool}, x::Real) = Quaternion(q.re - x, q.im, q.jm, q.km)
*(x::Real, q::Quaternion{Bool}) = Quaternion(x * q.re, x * q.im, x * q.jm, x * q.km)
*(q::Quaternion{Bool}, x::Real) = Quaternion(real(z) * x, imag(z) * x, q.jm * x, q.km * x)

+(x::Real, q::Quaternion) = Quaternion(x + q.re, q.im, q.jm, q.km)
+(q::Quaternion, x::Real) = Quaternion(x + q.re, q.im, q.jm, q.km)
+(z::Complex, q::Quaternion) = Quaternion(real(z) + q.re, imag(z) + q.im, q.jm, q.km)
+(q::Quaternion, z::Complex) = Quaternion(real(z) + q.re, imag(z) + q.im, q.jm, q.km)
function -(x::Real, q::Quaternion)
    # we don't want the default type for -(Bool)
    re = x - q.re
    Quaternion(re, - oftype(re, q.im), - oftype(re, q.jm), - oftype(re, q.km))
end
function -(z::Complex, q::Quaternion)
    # we don't want the default type for -(Bool)
    re = real(z) - q.re
    Quaternion(re, imag(z) - q.im, - oftype(re, q.jm), - oftype(re, q.km))
end
-(q::Quaternion, x::Real) = Quaternion(q.re - x, q.im, q.jm, q.km)
-(q::Quaternion, z::Complex) = Quaternion(q.re - real(z), q.im - imag(z), q.jm, q.km)
*(x::Real, q::Quaternion) = Quaternion(x * q.re, x * q.im, x * q.jm, x * q.km)
*(q::Quaternion, x::Real) = Quaternion(x * q.re, x * q.im, x * q.jm, x * q.km)

*(q::Quaternion, z::Complex) = Quaternion(q.re*z.re - q.im*z.im, q.re*z.im + q.im*z.re,
                                                      q.jm*z.re + q.km*z.im, q.km*z.re - q.jm*z.im)
*(z::Complex, q::Quaternion) = Quaternion(z.re*q.re - z.im*q.im, z.re*q.im + z.im*q.re,
                                                      z.re*q.jm - z.im*q.km, z.re*q.km + z.im*q.jm)

/(a::R, q::S) where {R<:Real,S<:Quaternion} = (T = promote_type(R,S); a*inv(T(q)))
/(q::Quaternion, x::Real) = Quaternion(q.re/x, q.im/x, q.jm/x, q.km/x)
/(q::Quaternion, w::Quaternion) = q * inv(w)

rand(r::AbstractRNG, ::Type{Quaternion{T}}) where {T<:Real} = Quaternion(rand(r,T), rand(r,T), rand(r,T), rand(r,T))
randn(r::AbstractRNG, ::Type{Quaternion{T}}) where {T<:Real} = Quaternion(randn(r,T), randn(r,T), randn(r,T), randn(r,T))

normalize(q::Quaternion{T}) where {T<:Real} = Quaternion(normalize(vec(q)))

exp(q::Quaternion{<:Integer}) = exp(float(q))
log(q::Quaternion{<:Integer}) = log(float(q))

function exp(q::Quaternion{<:AbstractFloat})
    V = vecnorm([q.im q.jm q.km])
    exp(q.re)*(cos(V) + quat(zero(q.re), q.im/V, q.jm/V, q.km/V)*sin(V))
end

function log(q::Quaternion{<:AbstractFloat})
    V = vecnorm([q.im q.jm q.km])
    Q = abs(q)
    log(Q) + quat(zero(q.re), q.im/V, q.jm/V, q.km/V)*acos(q.re/Q)
end

sqrt(q::Quaternion) = exp(0.5*log(q))
^(q::Quaternion, w::Quaternion) = exp(w*log(q))

function round(q::Quaternion{<:AbstractFloat}, ::RoundingMode{MR}, ::RoundingMode{MI}) where {MR,MI}
    Quaternion(round(q.re, RoundingMode{MR}()),
               round(q.im, RoundingMode{MI}()),
               round(q.jm, RoundingMode{MI}()),
               round(q.km, RoundingMode{MI}()))
end

round(q::Quaternion) = Quaternion(round(q.re), round(q.im), round(q.jm), round(q.km))

function round(q::Quaternion, digits::Integer, base::Integer=10)
    Complex(round(q.re, digits, base),
            round(q.im, digits, base),
            round(q.jm, digits, base),
            round(q.km, digits, base))
end

float(q::Quaternion{<:AbstractFloat}) = q
float(q::Quaternion) = Quaternion(float(q.re), float(q.im), float(q.jm), float(q.km))

big(q::Quaternion{<:AbstractFloat}) = Quaternion{BigFloat}(q)
big(q::Quaternion{<:Integer}) = Quaternion{BigInt}(q)


## Matrix representations of Quaternions

function cmatrix(q::Quaternion)
    [complex( q.re, q.im) complex(q.jm,  q.km);
     complex(-q.jm, q.km) complex(q.re, -q.im)] 
end

function cmatrix(Q::Matrix{Quaternion{T}}) where {T}
    W = zeros(Complex{T}, 2 .* size(Q))
    for i=1:size(Q, 1), j=1:size(Q, 2)
        W[2i-1:2i, 2j-1:2j] = cmatrix(Q[i, j])
    end
    W
end

function qmatrix(C::Matrix{Complex{T}}) where {T}
    Q = zeros(Quaternion{T}, Integer.(size(C)./2))
    for i=1:size(Q, 1), j=1:size(Q, 2)
        Q[i,j] = real(C[2i-1, 2j-1]) + imag(C[2i-1, 2j-1])*im + real(C[2i-1, 2j])*jm + imag(C[2i-1, 2j])*km
    end
    Q
end

## Array operations on complex numbers ##

quat(A::AbstractArray{<:Quaternion}) = A


_default_type(T::Type{Quaternion}) = Quaternion{Float}

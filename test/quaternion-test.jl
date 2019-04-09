
@testset "quat basic tests" begin

    @testset "type conversion" for T in (Int32, Int64, Float32, Float64, BigFloat, BigInt)
        @test real(quat(T)) == T
        @test complex(quat(T)) == complex(T)
        @test widen(Quaternion{T}) == Quaternion{widen(T)}
        @test quat(Complex{T}) == Quaternion{T}
        @test quat(Quaternion{T}) == Quaternion{T}
    end

    @test promote_rule(QuatF16, Float32) == QuatF32
    @test promote_rule(QuatF64, ComplexF32) == QuatF64
    @test promote_rule(Quaternion{BigFloat}, QuatF64) == Quaternion{BigFloat}



    @test real(quat(1.0)) == 1.0
    @test real(quat(1.0+2.0im)) == 1.0
    @test convert(Quaternion, 1.0) == quat(1.0)
    @test convert(Quaternion, 1.0+2.0im) == quat(1.0+2.0im)
    @test convert(Quaternion, 1+im+jm+km) == quat(1,1,1,1)
    @test convert(Quaternion{Float32}, 1.0) == quat(1.0)
    @test convert(Quaternion{Float32}, 1.0+2.0im) == quat(1.0+2.0im)
    @test convert(Quaternion{Int32}, 1+im+jm+km) == quat(1,1,1,1)
    @test imag(quat(1.0+2.0im)) == 2.0
    @test jmag(1.0+2.0im+3.0jm+4.0km) == 3.0
    @test kmag(1.0+2.0im+3.0jm+4.0km) == 4.0
    @test quat(vec([0.0 2.0 3.0 4.0])) == 2im + 3jm + 4km
    @test isinteger(quat(1,0,0,0))
    @test !isinteger(quat(1.1, 0.0, 0.0, 0.0))
    @test !isinteger(jm)
    @test isfinite(quat(1.0, 10.0, 100.0, 1000.0))
    @test !isfinite(quat(1.0, Inf, Inf, Inf))
    @test isnan(quat(NaN, NaN, NaN, NaN))
    @test !isnan(1 + im + jm + km)
    @test isinf(quat(1.0, Inf, Inf, Inf))
    @test !isinf(quat(1.0, 10.0, 100.0, 1000.0))
    @test iszero(quat(0.0, 0.0, 0.0, 0.0))
    @test iszero(quat(0, 0, 0, 0))
    @test !iszero(im + jm + km)
    @test quat(jm+km) == jm+km
    @test complex(1+im+jm+km) == 1+im
    @test vec(1+2im+3jm+4km) == [1; 2; 3; 4]

    @test conj(1+2im+3jm+4km) == 1-2im-3jm-4km

    @test quat(false,true,false,false) == im
    @test quat(false,false,true,false) == jm
    @test quat(false,false,false,true) == km
    @test quat(1,0,0,0) == 1
    @test quat(0,1,0,0) == im
    @test quat(0,0,1,0) == jm
    @test quat(0,0,0,1) == km
    @test quat(1.0,0.0,0.0,0.0) == 1
    @test quat(0.0,1.0,0.0,0.0) == im
    @test quat(0.0,0.0,1.0,0.0) == jm
    @test quat(0.0,0.0,0.0,1.0) == km
    @test quat(-1,0,0,0) == -1
    @test quat(0,-1,0,0) == -im
    @test quat(0,0,-1,0) == -jm
    @test quat(0,0,0,-1) == -km
    @test quat(-1.0,0.0,0.0,0.0) == -1
    @test quat(0.0,-1.0,0.0,0.0) == -im
    @test quat(0.0,0.0,-1.0,0.0) == -jm
    @test quat(0.0,0.0,0.0,-1.0) == -km
    @test im*jm ==  km == -jm*im
    @test jm*im == -km == -im*jm
    @test jm*km ==  im == -km*jm
    @test km*jm == -im == -jm*km
    @test km*im ==  jm == -im*km
    @test im*km == -jm == -km*im
    @test im*im == jm*jm == km*km == -1

    @test quat(complex(1,2), complex(3,4)) == quat(1,2,3,4)
    @test quat(quat(1,2,3,4)) == quat(1,2,3,4)
    @test quat(jm) == jm
    @test Quaternion(km) == km
end

# These mirror tests from Complex.jl
@testset "arithmetic" begin
    @testset for T in (Float16, Float32, Float64, BigFloat) # BigFloat was failing
        t = true
        f = false
        u = quat(T(+1.0), T(+1.0), T(+1.0), T(+1.0))

        @testset "add and subtract" begin

            @test isequal(T(+0.0) + im + jm + km, quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(T(-0.0) + im + jm + km, quat(T(-0.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(T(+0.0) - im - jm - km, quat(T(+0.0),T(-1.0),T(-1.0),T(-1.0)))
            @test isequal(T(-0.0) - im - jm - km, quat(T(-0.0),T(-1.0),T(-1.0),T(-1.0)))
            @test isequal(T(+1.0) + im + jm + km, quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(T(-1.0) + im + jm + km, quat(T(-1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(T(+1.0) - im - jm - km, quat(T(+1.0),T(-1.0),T(-1.0),T(-1.0)))
            @test isequal(T(-1.0) - im - jm - km, quat(T(-1.0),T(-1.0),T(-1.0),T(-1.0)))
            @test isequal(jm + T(+0.0), quat(T(+0.0),T(+0.0),T(+1.0),T(+0.0)))
            @test isequal(jm + T(-0.0), quat(T(-0.0),T(+0.0),T(+1.0),T(+0.0)))
            @test isequal(jm - T(+0.0), quat(T(+0.0),T(+0.0),T(+1.0),T(+0.0)))
            @test isequal(jm - T(-0.0), quat(T(+0.0),T(+0.0),T(+1.0),T(+0.0)))
            @test isequal(km + T(+0.0), quat(T(+0.0),T(+0.0),T(+0.0),T(+1.0)))
            @test isequal(km + T(-0.0), quat(T(-0.0),T(+0.0),T(+0.0),T(+1.0)))
            @test isequal(km - T(+0.0), quat(T(+0.0),T(+0.0),T(+0.0),T(+1.0)))
            @test isequal(km - T(-0.0), quat(T(+0.0),T(+0.0),T(+0.0),T(+1.0)))
            @test isequal(im + jm + km + T(+1.0), quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(im + jm + km + T(-1.0), quat(T(-1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(im + jm + km - T(+1.0), quat(T(-1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(im + jm + km - T(-1.0), quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test T(f) + im + jm + km == quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0))
            @test T(t) + im + jm + km == quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0))
            @test T(f) - im - jm - km == quat(T(+0.0),T(-1.0),T(-1.0),T(-1.0))
            @test T(t) - im - jm - km == quat(T(+1.0),T(-1.0),T(-1.0),T(-1.0))
            @test im + jm + km + T(f) == quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0))
            @test im + jm + km + T(t) == quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0))
            @test im + jm + km - T(f) == quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0))
            @test im + jm + km - T(t) == quat(T(-1.0),T(+1.0),T(+1.0),T(+1.0))
        end

        @testset "multiply" begin
            oq  = quat(T(+1.0),T(+0.0),T(+0.0),T(+0.0))
            iq  = quat(T(+0.0),T(+1.0),T(+0.0),T(+0.0))
            jq  = quat(T(+0.0),T(+0.0),T(+1.0),T(+0.0))
            kq  = quat(T(+0.0),T(+0.0),T(+0.0),T(+1.0))
            oqn = quat(T(-1.0),T(-0.0),T(-0.0),T(-0.0))
            iqn = quat(T(-0.0),T(-1.0),T(-0.0),T(-0.0))
            jqn = quat(T(-0.0),T(-0.0),T(-1.0),T(-0.0))
            jkn = quat(T(-0.0),T(-0.0),T(-0.0),T(-1.0))
            @test isequal(T(+1.0) * u,  u)
            @test isequal(u * T(+1.0),  u)
            @test isequal(T(-1.0) * u, -u)
            @test isequal(u * T(-1.0), -u)
            @test isequal(T(+0.0) * u, quat(T(+0.0),T(+0.0),T(+0.0),T(+0.0)))
            @test isequal(u * T(+0.0), quat(T(+0.0),T(+0.0),T(+0.0),T(+0.0)))
            @test isequal(T(-0.0) * u, quat(T(-0.0),T(-0.0),T(-0.0),T(-0.0)))
            @test isequal(u * T(-0.0), quat(T(-0.0),T(-0.0),T(-0.0),T(-0.0)))
            @test isequal(oq * oq, oq)
            @test isequal(iq * iq, quat(T(-1.0),T(+0.0),T(+0.0),T(+0.0)))
            @test isequal(jq * jq, quat(T(-1.0),T(+0.0),T(+0.0),T(+0.0)))
            @test isequal(kq * kq, quat(T(-1.0),T(+0.0),T(+0.0),T(+0.0)))
            @test iq*jq ==  kq == -jq*iq
            @test jq*iq == -kq == -iq*jq
            @test jq*kq ==  iq == -kq*jq
            @test kq*jq == -iq == -jq*kq
            @test kq*iq ==  jq == -iq*kq
            @test iq*kq == -jq == -kq*iq
        end

        @testset "divide" begin
            @test u/T(+1.0) == u
            @test u/quat(T(+1.0)) == u
            @test T(+1.0)/u == 0.25conj(u)
        end
    end

end

@testset "printing" begin
    @test sprint(show,  1 + 2im + 3jm + 4km) == "1 + 2im + 3jm + 4km"
    @test sprint(show, -1 - 2im - 3jm - 4km) == "-1 - 2im - 3jm - 4km"
    @test sprint(show,  1.0+2.0im+3.0jm+4.0km, context=:compact=>true) == "1.0+2.0im+3.0jm+4.0km"
    @test sprint(show, -1.0-2.0im-3.0jm-4.0km, context=:compact=>true) == "-1.0-2.0im-3.0jm-4.0km"
    @test sprint(show, NaN + NaN*im + NaN*jm + NaN*km) == "NaN + NaN*im + NaN*jm + NaN*km"
    @test sprint(show, quat(im)) == "im"
    @test sprint(show, jm) == "jm"
    @test sprint(show, km) == "km"
    @test sprint(show, Quaternion(true,true,true,true)) == "Quaternion(true,true,true,true)"
end

# @testset "exp(q)" begin


# @testset "Matrix tests" begin

#     Q = [1+2im+3jm+4jm -im+km; -jm+km -5+6im-7jm-8km]

#     @test qmatrix(cmatrix(1+2im+3jm+4km))[1,1] == 1+2im+3jm+4km
#     @test qmatrix(cmatrix(Q)) == Q

#     @test quat(Q) == Q
# end

# end

@testset "math" begin

    @test abs(quat(1,1,1,1)) == 2.0
    @test abs(quat(-1.0,2.0,-3.0,4.0)) == sqrt(1+4+9+16)
    @test abs2(quat(1,1,1,1)) == 4
    @test abs2(quat(-1.0,2.0,-3.0,4.0)) == 1.0+4.0+9.0+16.0

    @test inv(quat(1.0,0,0,0)) == quat(1.0,0,0,0)
    @test inv(quat(1,0,0,0)) == quat(1.0,0,0,0)
    @test inv(quat(0,1,0,0)) == quat(-1.0im)
    @test inv(jm) == -1.0jm
    @test inv(4km) == -0.25km
end

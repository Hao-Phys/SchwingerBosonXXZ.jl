@testmodule ExternalFieldTestHelpers begin
    using SchwingerBosonXXZ

    const SB = SchwingerBosonXXZ

    function field_test_system(; kwargs...)
        return SchwingerBosonSystem(
            1.0,
            1.0,
            0.5,
            0.4,
            2;
            kwargs...,
        )
    end

    function field_test_dynamical_matrix(sbs)
        D = zeros(ComplexF64, 12, 12)
        SB.dynamical_matrix!(D, sbs, SB.Vec3(0.17, 0.31, 0.0))
        return D
    end
end

@testitem "External-field interface" setup=[ExternalFieldTestHelpers] begin
    using LinearAlgebra

    const SB = SchwingerBosonXXZ
    const field_test_system =
        ExternalFieldTestHelpers.field_test_system

    default_system = field_test_system()

    @test default_system.h_ext == 0.0
    @test default_system.h_ext_direction == SB.Vec3(0.0, 0.0, 1.0)

    tuple_system = field_test_system(
        h_ext = 0.25,
        h_ext_direction = (3.0, 0.0, 4.0),
    )

    @test tuple_system.h_ext == 0.25
    @test tuple_system.h_ext_direction ≈ SB.Vec3(0.6, 0.0, 0.8)
    @test norm(tuple_system.h_ext_direction) ≈ 1.0

    displayed = sprint(show, MIME"text/plain"(), tuple_system)
    @test contains(displayed, "h_ext = 0.25")
    @test contains(displayed, "h_ext_direction")

    vector_direction = [0.0, 2.0, 0.0]
    vector_system = field_test_system(
        h_ext = 0.5,
        h_ext_direction = vector_direction,
    )

    vector_direction[2] = 7.0
    @test vector_system.h_ext_direction == SB.Vec3(0.0, 1.0, 0.0)

    @test set_external_field!(vector_system, 0.75, [1.0, 1.0, 0.0]) ===
        vector_system
    @test vector_system.h_ext == 0.75
    @test vector_system.h_ext_direction ≈
        SB.Vec3(inv(sqrt(2.0)), inv(sqrt(2.0)), 0.0)

    old_field = vector_system.h_ext
    old_direction = vector_system.h_ext_direction

    @test_throws ArgumentError set_external_field!(
        vector_system,
        0.4,
        (0.0, 0.0, 0.0),
    )
    @test vector_system.h_ext == old_field
    @test vector_system.h_ext_direction == old_direction

    @test_throws ArgumentError field_test_system(h_ext = -0.1)
    @test_throws ArgumentError field_test_system(h_ext = Inf)
    @test_throws ArgumentError field_test_system(
        h_ext_direction = (1.0, 2.0),
    )
    @test_throws ArgumentError field_test_system(
        h_ext_direction = (0.0, 0.0, 0.0),
    )
    @test_throws ArgumentError field_test_system(
        h_ext_direction = (1.0, NaN, 0.0),
    )
end

@testitem "External-field BdG contribution" setup=[ExternalFieldTestHelpers] begin
    using LinearAlgebra

    const SB = SchwingerBosonXXZ
    const field_test_system =
        ExternalFieldTestHelpers.field_test_system
    const field_test_dynamical_matrix =
        ExternalFieldTestHelpers.field_test_dynamical_matrix

    zero_field = field_test_system()
    D_zero = field_test_dynamical_matrix(zero_field)

    zero_field_other_direction = field_test_system(
        h_ext = 0.0,
        h_ext_direction = [1.0, 1.0, 1.0],
    )

    @test field_test_dynamical_matrix(zero_field_other_direction) == D_zero

    for direction in (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, -2.0, 3.0),
    )
        sbs = field_test_system(
            h_ext = 0.3,
            h_ext_direction = direction,
        )
        D = field_test_dynamical_matrix(sbs)

        spin_matrix =
            -sbs.S * sbs.h_ext *
            sum(
                sbs.h_ext_direction[μ] * SB.σs[μ]
                for μ in 1:3
            )

        for sublattice in 1:3
            particle = 2sublattice-1:2sublattice
            hole = particle .+ 6

            @test D[particle, particle] ≈
                D_zero[particle, particle] + spin_matrix
            @test D[hole, hole] ≈
                D_zero[hole, hole] + transpose(spin_matrix)
        end

        @test ishermitian(D)
    end

    # A uniform external field is additive with the sublattice-dependent
    # symmetry-breaking field.
    h_SB = 0.2
    θs = [0.1, 0.7, 1.3]

    symmetry_breaking_only = field_test_system(; h_SB, θs)
    combined = field_test_system(
        ;
        h_SB,
        θs,
        h_ext = 0.3,
        h_ext_direction = (0.0, 1.0, 0.0),
    )

    D_SB = field_test_dynamical_matrix(symmetry_breaking_only)
    D_combined = field_test_dynamical_matrix(combined)
    external_spin_matrix = -combined.S * combined.h_ext * SB.σs[2]

    for sublattice in 1:3
        particle = 2sublattice-1:2sublattice
        hole = particle .+ 6

        @test D_combined[particle, particle] - D_SB[particle, particle] ≈
            external_spin_matrix
        @test D_combined[hole, hole] - D_SB[hole, hole] ≈
            transpose(external_spin_matrix)
    end
end

@testitem "External-field free-energy derivative" setup=[ExternalFieldTestHelpers] begin
    using LinearAlgebra

    const SB = SchwingerBosonXXZ
    const field_test_system =
        ExternalFieldTestHelpers.field_test_system

    direction = (1.0, -2.0, 3.0)
    sbs = field_test_system(
        h_ext = 0.15,
        h_ext_direction = direction,
    )
    set_μ0!(sbs, [-3.0, -3.0, -3.0])

    function fixed_saddle_bosonic_free_energy(sbs, h_ext)
        set_external_field!(sbs, h_ext)
        D = zeros(ComplexF64, 12, 12)
        V = zeros(ComplexF64, 12, 12)
        return SB.bosonic_free_energy!(nothing, V, D, sbs)
    end

    h0 = sbs.h_ext
    step = 1e-6
    derivative = (
        fixed_saddle_bosonic_free_energy(sbs, h0 + step) -
        fixed_saddle_bosonic_free_energy(sbs, h0 - step)
    ) / (2step)

    set_external_field!(sbs, h0)

    P = zeros(ComplexF64, 12, 12)
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    spin_vertex = zeros(ComplexF64, 12, 12)
    magnetization_along_field = Ref(0.0)

    for i in 1:sbs.L, j in 1:sbs.L
        q = SB.Vec3((i - 1) / sbs.L, (j - 1) / sbs.L, 0.0)
        SB.single_particle_density_matrix!(P, D, V, tmp, sbs, q)

        for sublattice in 1:3, μ in 1:3
            SB.∂ID∂S!(spin_vertex, sublattice, μ, sbs)
            magnetization_along_field[] +=
                sbs.h_ext_direction[μ] * real(tr(P * spin_vertex)) / sbs.L^2
        end
    end

    @test derivative ≈ -magnetization_along_field[] atol = 1e-7 rtol = 1e-6
end


function divided_difference!(sbs, Dmat, E; atol=1e-8)
    Dmat .= 0.0
    T = sbs.T
    for i in eachindex(E), j in eachindex(E)
        if isapprox(E[i], E[j]; atol)
            Dmat[i, j] += T > 1e-8 ? -csch(E[i]/(2T))^2 / (2T) : 0.0
        else
            Dmat[i, j] += (coth(E[i]/2T) - coth(E[j]/2T)) / (E[i] - E[j])
        end
    end
end


function jacobian!(sbs::SchwingerBosonSystem, x, ja)
    ja .= 0.0
    set_x!(sbs, x)

    D = zeros(ComplexF64, 12, 12)
    Dmat = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    tmp2 = zeros(ComplexF64, 12, 12)

    (; L) = sbs
    Nu = L^2

    ∂D∂X_re = zeros(ComplexF64, 12, 12)
    ∂D∂X_im = zeros(ComplexF64, 12, 12)
    ∂D∂la = zeros(ComplexF64, 12, 12)

    ja_mat = reshape(ja, 4, 27)

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)
        inv_V = inv(V)
        divided_difference!(sbs, Dmat, E)
        for α in 1:3
            # Calculate ∂ (ID) / ∂λα
            ∂D∂λ!(∂D∂la, α)
            mul!(tmp, Ĩ, ∂D∂la)
            copyto!(∂D∂la, tmp)

            for β in 1:3
                ∂D∂A!(∂D∂X_re, ∂D∂X_im, sbs, q, β)

                mul!(tmp, Ĩ, ∂D∂X_re)
                copyto!(∂D∂X_re, tmp)
                mul!(tmp, inv_V, ∂D∂X_re)
                mul!(∂D∂X_re, tmp, V)
                tmp .= Dmat .* ∂D∂X_re
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β] += real(tr(tmp * ∂D∂la)) / (4Nu)

                mul!(tmp, Ĩ, ∂D∂X_im)
                copyto!(∂D∂X_im, tmp)
                mul!(tmp, inv_V, ∂D∂X_im)
                mul!(∂D∂X_im, tmp, V)
                tmp .= Dmat .* ∂D∂X_im
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β+12] += real(tr(tmp * ∂D∂la)) / (4Nu)

                ∂D∂B!(∂D∂X_re, ∂D∂X_im, sbs, q, β)

                mul!(tmp, Ĩ, ∂D∂X_re)
                copyto!(∂D∂X_re, tmp)
                mul!(tmp, inv_V, ∂D∂X_re)
                mul!(∂D∂X_re, tmp, V)
                tmp .= Dmat .* ∂D∂X_re
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β+3] += real(tr(tmp * ∂D∂la)) / (4Nu)

                mul!(tmp, Ĩ, ∂D∂X_im)
                copyto!(∂D∂X_im, tmp)
                mul!(tmp, inv_V, ∂D∂X_im)
                mul!(∂D∂X_im, tmp, V)
                tmp .= Dmat .* ∂D∂X_im
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β+15] += real(tr(tmp * ∂D∂la)) / (4Nu)

                ∂D∂C!(∂D∂X_re, ∂D∂X_im, sbs, q, β)

                mul!(tmp, Ĩ, ∂D∂X_re)
                copyto!(∂D∂X_re, tmp)
                mul!(tmp, inv_V, ∂D∂X_re)
                mul!(∂D∂X_re, tmp, V)
                tmp .= Dmat .* ∂D∂X_re
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β+6] += real(tr(tmp * ∂D∂la)) / (4Nu)

                mul!(tmp, Ĩ, ∂D∂X_im)
                copyto!(∂D∂X_im, tmp)
                mul!(tmp, inv_V, ∂D∂X_im)
                mul!(∂D∂X_im, tmp, V)
                tmp .= Dmat .* ∂D∂X_im
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β+18] += real(tr(tmp * ∂D∂la)) / (4Nu)

                ∂D∂D!(∂D∂X_re, ∂D∂X_im, sbs, q, β)

                mul!(tmp, Ĩ, ∂D∂X_re)
                copyto!(∂D∂X_re, tmp)
                mul!(tmp, inv_V, ∂D∂X_re)
                mul!(∂D∂X_re, tmp, V)
                tmp .= Dmat .* ∂D∂X_re
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β+9] += real(tr(tmp * ∂D∂la)) / (4Nu)

                mul!(tmp, Ĩ, ∂D∂X_im)
                copyto!(∂D∂X_im, tmp)
                mul!(tmp, inv_V, ∂D∂X_im)
                mul!(∂D∂X_im, tmp, V)
                tmp .= Dmat .* ∂D∂X_im
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β+21] += real(tr(tmp * ∂D∂la)) / (4Nu)

                ∂D∂λ!(∂D∂X_re, β)

                mul!(tmp, Ĩ, ∂D∂X_re)
                copyto!(∂D∂X_re, tmp)
                mul!(tmp, inv_V, ∂D∂X_re)
                mul!(∂D∂X_re, tmp, V)
                tmp .= Dmat .* ∂D∂X_re
                mul!(tmp2, V, tmp)
                mul!(tmp, tmp2, inv_V)
                ja_mat[α, β+24] += real(tr(tmp * ∂D∂la)) / (4Nu)
            end

        end

    end

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        inv_D = inv(D)
        for α in 1:3
            ∂D∂A!(∂D∂X_re, ∂D∂X_im, sbs, q, α)
            ja_mat[4, α] += real(tr(inv_D * ∂D∂X_re)) / Nu
            ja_mat[4, α+12] += real(tr(inv_D * ∂D∂X_im)) / Nu
            ∂D∂B!(∂D∂X_re, ∂D∂X_im, sbs, q, α)
            ja_mat[4, α+3] += real(tr(inv_D * ∂D∂X_re)) / Nu
            ja_mat[4, α+15] += real(tr(inv_D * ∂D∂X_im)) / Nu
            ∂D∂C!(∂D∂X_re, ∂D∂X_im, sbs, q, α)
            ja_mat[4, α+6] += real(tr(inv_D * ∂D∂X_re)) / Nu
            ja_mat[4, α+18] += real(tr(inv_D * ∂D∂X_im)) / Nu
            ∂D∂D!(∂D∂X_re, ∂D∂X_im, sbs, q, α)
            ja_mat[4, α+9] += real(tr(inv_D * ∂D∂X_re)) / Nu
            ja_mat[4, α+21] += real(tr(inv_D * ∂D∂X_im)) / Nu
            ∂D∂λ!(∂D∂X_re, α)
            ja_mat[4, α+24] += real(tr(inv_D * ∂D∂X_re)) / Nu
        end
    end
end
# Given q_reshaped in reciprocal lattice units (RLU) for the possibly-reshaped crystal, return a
# q in RLU for the original crystal.
to_original_rlu(q) = recipvecs_origin \ (recipvecs_reduce * q) 

struct ReshapedQResult
    q_reshaped_closest :: Vec3
    q_index :: CartesianIndex{3}
end

struct QPath
    qs :: Vector{Vec3}
    xticks :: Tuple{Vector{Int64}, Vector{String}}
end

# Given `q` in reciprocal lattice units (RLU) for the original crystal and the `sbs` object, return to the `ReshapedQResult` which contains:
# - `q_reshaped_closest`: the reshaped momentum in RLU that is closest to the input momentum
# - `q_index`: the Cartesian index of the closest momentum in the finite-size grid
function to_reshaped_q_sbs(sbs::SchwingerBosonSystem, q)
    (; L) = sbs
    qs = [Vec3(i/L, j/L, 0.0) for i in 0:L-1, j in 0:L-1, _ in 1:1]

    # Here we mod one. This is because the q_reshaped is in the reciprocal lattice unit, and we need to find the closest q in the grid.
    q_reshaped = to_reshaped_rlu(q)
    for i in 1:3
        (abs(q_reshaped[i]) < 1e-12) && (q_reshaped = setindex(q_reshaped, 0.0, i))
    end
    # Fold the reshaped wave vector within in first magnetic Brillouin zone
    q_reshaped_folded = mod.(q_reshaped, 1.0)
    G_mag = q_reshaped - q_reshaped_folded
    for i in 1:3
        (abs(q_reshaped_folded[i]) < 1e-12) && (q_reshaped_folded = setindex(q_reshaped_folded, 0.0, i))
    end
    norm_diff, q_index = findmin(x -> norm(x - q_reshaped_folded), qs)

    q_reshaped_closest = qs[q_index] + G_mag

    if norm_diff > 1e-12
        q_closest = to_original_rlu(q_reshaped_closest)
        Δq = norm(recipvecs_origin * (q - q_closest))
        @warn "The requested momentum $q is not available in the set of `qs` used for the NPT calculation. The closest available momentum $q_closest is used instead (‖Δq‖ = $Δq)."
    end

    return ReshapedQResult(q_reshaped_closest, q_index)
end

function q_space_path_sbs(sbs::SchwingerBosonSystem, qs; labels=nothing)
    (; L) = sbs
    clustersize = (L, L, 1)

    reshaped_q_res = [to_reshaped_q_sbs(sbs, q) for q in qs]
    length_qs = length(qs)

    path = Vec3[]
    markers = Int[]

    for i in 1:length_qs - 1
        push!(markers, length(path)+1)
        q_reshaped_s = reshaped_q_res[i].q_reshaped_closest
        q_reshaped_e = reshaped_q_res[i+1].q_reshaped_closest
        Δq_reshaped = q_reshaped_e - q_reshaped_s
        Δns = round.(Int, abs.(Δq_reshaped .* collect(clustersize)))
        Δn = gcd(gcd(Δns[1], Δns[2]), Δns[3])
        j_end = i == length_qs - 1 ? Δn : Δn - 1
        for j in 0:j_end
            q_reshaped = q_reshaped_s + j/Δn * Δq_reshaped
            q = to_original_rlu(q_reshaped)
            push!(path, q)
        end
    end

    push!(markers, length(path))

    labels = @something labels vec3_to_string.(qs)
    xticks = (markers, labels)
    return QPath(path, xticks)
end
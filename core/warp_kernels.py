# pyright: reportInvalidTypeForm=false
import warp as wp
import numpy as np


@wp.func
def poly6(r: wp.float64, h: wp.float64) -> wp.float64:
    """Poly6 smoothing kernel."""
    if r < wp.float64(0.0) or r >= h:
        return wp.float64(0.0)
      
    term = h * h - r * r
    return (wp.float64(4.0) / (wp.float64(wp.pi) * (h ** wp.float64(8.0)))) * (term * term * term)

@wp.func
def w_spiky(r: float, h: float) -> float:
    # Matched CPU using h^2 instead of h^6. Pure 32-bit.
    if r < h:
        return (45.0 / (wp.pi * (h * h))) * (h - r) * (h - r)
    return 0.0


@wp.func
def color_weight_pressure(ci: int, cj: int) -> float:
    """Pressure weight: 1.0 same color, 0.0 cross-color.
    Clips negative cell_index (fixed cells) to 0, matching CPU np.maximum(0, cell_index)."""
    ci_c = wp.max(ci, 0)
    cj_c = wp.max(cj, 0)
    if ci_c == cj_c:
        return 1.0
    return 0.0


@wp.func
def color_weight_curvature(ci: int, cj: int) -> float:
    """Curvature weight: 1.0 same color, 0.1 cross-color.
    Clips negative cell_index (fixed cells) to 0, matching CPU np.maximum(0, cell_index)."""
    ci_c = wp.max(ci, 0)
    cj_c = wp.max(cj, 0)
    if ci_c == cj_c:
        return 1.0
    return 0.1


@wp.func
def pairwise_scalar(d_i: float, d_j: float, r: float, h: float, rho_target: float, wc: float) -> float:
    ws = w_spiky(r, h)
    denom_i = d_i * d_i
    denom_j = d_j * d_j
    
    if denom_i < 1e-6:
        denom_i = 1e-6
    if denom_j < 1e-6:
        denom_j = 1e-6
    scalar = ((d_i - rho_target) / denom_i) + ((d_j - rho_target) / denom_j)
    return scalar * ws * wc


@wp.func
def proj_vec(v: wp.vec3, n: wp.vec3) -> wp.vec3:
    """Project vector v onto unit normal n."""
    dot = v.x * n.x + v.y * n.y + v.z * n.z
    return wp.vec3(dot * n.x, dot * n.y, dot * n.z)


@wp.func
def proj_vec_64(v: wp.vec3d, n: wp.vec3d) -> wp.vec3d:
    dot = v.x * n.x + v.y * n.y + v.z * n.z
    return wp.vec3d(dot * n.x, dot * n.y, dot * n.z)

# keep high-precision wp_positions (64-bit) for integration and volume calculation, 
# but downcast to 32-bit for neighbor queries and force calculations to save memory and bandwidth.
@wp.kernel
def downcast_positions_kernel(
    pos_64: wp.array(dtype=wp.vec3d),
    pos_32: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    p = pos_64[tid]
    pos_32[tid] = wp.vec3(float(p.x), float(p.y), float(p.z))

@wp.kernel
def compute_density_kernel(
    positions_64: wp.array(dtype=wp.vec3d), # EXACT 64-bit array
    positions_32: wp.array(dtype=wp.vec3),  # 32-bit grid array
    grid: wp.uint64,
    h: float,
    wp_densities: wp.array(dtype=float),
    neighbor_counts: wp.array(dtype=int),
    phase_counts: wp.array(dtype=int),     
    cell_index: wp.array(dtype=int),
):
    tid = wp.tid()
    pos_i_32 = positions_32[tid]
    pos_i_64 = positions_64[tid]
    
    query = wp.hash_grid_query(grid, pos_i_32, h)
    neighbor_index = int(0)

    accumulated_value = poly6(wp.float64(0.0), wp.float64(h)) 
    count = int(0)
    diff_count = int(0)               

    while wp.hash_grid_query_next(query, neighbor_index):
        if neighbor_index != tid:
            pos_j_64 = positions_64[neighbor_index]
            
            # EXACT 64-bit distance calculation prevents edge-case boundary misses
            r = wp.length(pos_i_64 - pos_j_64)
            
            if r < wp.float64(h + 1e-6):
                w = poly6(r, wp.float64(h)) 
                accumulated_value += w
                
                ci = wp.max(cell_index[tid], 0)
                cj = wp.max(cell_index[neighbor_index], 0)
                if ci == cj:
                    count += 1
                else:
                    diff_count += 1       

    density_factor = (h * h) / 4.0
    wp_densities[tid] = float(accumulated_value) * density_factor
    neighbor_counts[tid] = count
    phase_counts[tid] = diff_count

@wp.kernel
def compute_forces_kernel(
    positions_64: wp.array(dtype=wp.vec3d), # EXACT 64-bit array
    positions_32: wp.array(dtype=wp.vec3),  # 32-bit grid array
    densities: wp.array(dtype=float),
    grid: wp.uint64,
    h: float,
    k_plane: float,
    k_dist: float,
    cell_index: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    out_forces: wp.array(dtype=wp.vec3),
    out_curvature: wp.array(dtype=wp.vec3),
    rho_target: float
):
    tid = wp.tid()
    pos_i_32 = positions_32[tid]
    pos_i_64 = positions_64[tid]

    query = wp.hash_grid_query(grid, pos_i_32, h)
    neighbor_index = int(0)
    
    # 64-bit accumulators prevent catastrophic cancellation when summing tiny vectors
    numerator = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    denom = wp.float64(1.0)  
    pair_force = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    
    pressure_i = densities[tid]
    
    while wp.hash_grid_query_next(query, neighbor_index):
        if neighbor_index != tid:
            pos_j_64 = positions_64[neighbor_index]
            delta = pos_j_64 - pos_i_64
            r = wp.length(delta) # 64-bit length
            
            if r < wp.float64(h + 1e-6) and r > wp.float64(1e-10):
                weight = wp.float64(1.0) - (r / wp.float64(h))
                wc_curv = wp.float64(color_weight_curvature(cell_index[tid], cell_index[neighbor_index]))
                
                numerator = numerator + delta * (weight * wc_curv)
                denom = denom + weight
                
                wc_p = float(color_weight_pressure(cell_index[tid], cell_index[neighbor_index]))
                d_j = densities[neighbor_index]

                # Cast down to 32-bit strictly for the scalar empirical formula
                scalar = pairwise_scalar(pressure_i, d_j, float(r), h, rho_target, wc_p)
                
                # Apply scalar in 64-bit to prevent vector jitter
                pair_force = pair_force + (-wp.float64(scalar)) * (delta / r)

    # Finalize curvature precisely
    v_weighted = numerator / denom
    result_curv_64 = v_weighted * wp.float64(20.0 / 3.0) / wp.float64(h * h)

    # Upgrade normal to 64-bit for perfect geometric projection
    n_i = normals[tid]
    n_i_64 = wp.vec3d(wp.float64(n_i.x), wp.float64(n_i.y), wp.float64(n_i.z))

    result_curv_64 = proj_vec_64(result_curv_64, n_i_64)
    pair_force = pair_force - proj_vec_64(pair_force, n_i_64)

    # Downcast exactly once before writing to global memory
    res_curv_32 = wp.vec3(float(result_curv_64.x), float(result_curv_64.y), float(result_curv_64.z))
    res_pair_32 = wp.vec3(float(pair_force.x), float(pair_force.y), float(pair_force.z))
    
    out_curvature[tid] = res_curv_32
    out_forces[tid] = out_forces[tid] + (res_curv_32 * k_plane + res_pair_32 * k_dist)

@wp.kernel
def integrate_positions_kernel(
    positions: wp.array(dtype=wp.vec3d),
    forces_gpu: wp.array(dtype=wp.vec3),
    forces_cpu: wp.array(dtype=wp.vec3),
    cell_index: wp.array(dtype=int),
    dt: float,
    seed: int
):
    tid = wp.tid()
    
    if cell_index[tid] != -1:
        total_force = forces_gpu[tid] + forces_cpu[tid]
        disp = total_force * dt
        
        # Cast the 32-bit displacement to 64-bit
        disp_64 = wp.vec3d(wp.float64(disp.x), wp.float64(disp.y), wp.float64(disp.z))
        
        positions[tid] = positions[tid] + disp_64
        
        # Generate on-GPU Brownian noise (PDF Eq 107)
        #state = wp.rand_init(seed, tid)
        #noise_x = wp.randn(state) * 0.001 * wp.sqrt(dt)
        #noise_y = wp.randn(state) * 0.001 * wp.sqrt(dt)
        #noise_z = wp.randn(state) * 0.001 * wp.sqrt(dt)
        #noise = wp.vec3(noise_x, noise_y, noise_z)

        #positions[tid] = positions[tid] + (total_force * dt) + noise
        
        
@wp.kernel
def compute_centroids_kernel(
    positions: wp.array(dtype=wp.vec3d),
    cell_index: wp.array(dtype=int),
    cell_centroids: wp.array(dtype=wp.vec3d),
    cell_counts: wp.array(dtype=int)
):
    tid = wp.tid()
    c = cell_index[tid]
    # Skip fixed/boundary surfels (-1)
    if c >= 0:
        wp.atomic_add(cell_centroids, c, positions[tid])
        wp.atomic_add(cell_counts, c, 1)

@wp.kernel
def normalize_centroids_kernel(
    cell_centroids: wp.array(dtype=wp.vec3d),
    cell_counts: wp.array(dtype=int)
):
    c = wp.tid()
    if cell_counts[c] > 0:
        # cast integer count to float64 for division
        count_f64 = wp.float64(float(cell_counts[c]))
        cell_centroids[c] = cell_centroids[c] / count_f64

@wp.kernel
def compute_volumes_kernel(
    positions: wp.array(dtype=wp.vec3d),        
    normals: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    cell_index: wp.array(dtype=int),
    cell_centroids: wp.array(dtype=wp.vec3d), 
    cell_volumes: wp.array(dtype=wp.float64)   
):
    tid = wp.tid()
    c = cell_index[tid]
    if c >= 0:
        d_i = densities[tid]
        if d_i > 1e-6:
            surfel_area = wp.float64(1.0) / wp.float64(d_i)
            dist = positions[tid] - cell_centroids[c]
            r = wp.length(dist)
            
            if r > wp.float64(1e-6):
                # Normal is float32, need to cast distance back to float32 for the dot product.
                n64 = wp.vec3d(wp.float64(normals[tid].x), wp.float64(normals[tid].y), wp.float64(normals[tid].z))
                area_proj = surfel_area * wp.dot(dist / r, n64)
                vol = (area_proj * r) / wp.float64(3.0)
                wp.atomic_add(cell_volumes, c, vol)

@wp.kernel
def apply_volume_pressure_kernel(
    normals: wp.array(dtype=wp.vec3),
    cell_index: wp.array(dtype=int),
    cell_volumes: wp.array(dtype=wp.float64), 
    rest_volume: float,
    k_pressure: float,
    out_forces: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    c = cell_index[tid]
    if c >= 0:
        # Cast 64-bit volume back down to 32-bit math for pressure force
        vol = float(cell_volumes[c])
        pressure = 1.0 - (vol / rest_volume)
        force_vol = (k_pressure * pressure) * normals[tid]
        out_forces[tid] = out_forces[tid] + force_vol
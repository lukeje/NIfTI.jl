# reslice.jl
# Reslice one NIfTI volume to the space of another based on header information

using NIfTI
using ImageTransformations, CoordinateTransformations

# Round to an integer and replace NaN, Inf (and negative values) with 0 if the second argument is of (unsigned) integer type
mayberound{T<:Unsigned}(x, ::Type{T}) = (x ≥ 0) && isfinite(x) ? round(T, x) : 0
mayberound{T<:Integer}(x, ::Type{T}) = isfinite(x) ? round(T, x) : 0
mayberound(x, ::DataType) = x

function apply_rotation!(out, A, data)
    # apply 3D transformation separately for higher dimensions
    for idx in Iterators.product(axes(out)[4:end]...)
        out[:,:,:,idx...] = mayberound.(warp(data[:,:,:,idx...], A, axes(out)[1:3]), eltype(out))
    end
end

# Convert affine matrix from NIfTI header to format for `warp`; last term converts from 0- (NIfTI) and 1- (julia) based indexing
affineToCoordinateTransform(a) = Translation(a[1:3,4]) ∘ LinearMap(a[1:3,1:3]) ∘ Translation([-1, -1, -1])

function reslice{T<:FloatingPoint}(targ::NIVolume, mov::NIVolume{T}, outtype::DataType=T)
    # Construct the affine matrix that maps voxels of targ to voxels of mov
    targaffine = getaffine(targ.header)
    movaffine  = getaffine(mov.header)
    A = inv(affineToCoordinateTransform(movaffine)) ∘ affineToCoordinateTransform(targaffine)

    # Allocate output
    out = zeros(outtype, size(targ)[1:3]..., size(mov)[4:end]...)

    # Perform rotation
    apply_rotation!(out, A, mov.raw)

    # Set header of mov
    newheader = deepcopy(mov.header)
    setaffine(newheader, targaffine)
    newheader.pixdim = (newheader.pixdim[1], targ.header.pixdim[2:4]..., newheader.pixdim[5:end]...)
    NIVolume(newheader, mov.extensions, out)
end
reslice(targ::NIVolume, mov::NIVolume) = reslice(targ, NIVolume(mov.header, mov.extensions, map(Float32, mov.raw)), eltype(mov))
reslice(targ::AbstractString, mov::AbstractString, out::AbstractString) = niwrite(out, reslice(niread(targ, mmap=true), niread(mov, mmap=true)))

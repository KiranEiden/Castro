#ifndef problem_tagging_H
#define problem_tagging_H

///
/// Define problem-specific tagging criteria
///
/// @param i         x-index
/// @param j         y-index
/// @param k         z-index
/// @param tag       tag array (TagBox)
/// @param state     simulation state (Fab)
/// @param level     AMR level
/// @param geomdata  geometry data
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_tagging(int i, int j, int k,
                     Array4<char> const& tag,
                     Array4<Real const> const& state,
                     int level, const GeometryData& geomdata) {


	if(state(i,j,k,UFX)/state(i,j,k,URHO) > 1.0e-6_rt)
	{
		if(level < problem::max_shock_tagging_level)
			tag(i,j,k) = TagBox::SET;
	}

}

#endif

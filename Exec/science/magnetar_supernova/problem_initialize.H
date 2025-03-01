
AMREX_INLINE
void problem_initialize()
{
    for(int i = 0; i < AMREX_SPACEDIM; i++)
    {
        problem::center[i] = 0.0_rt;
    }
    
    if(problem::rand_seed > 0) amrex::ResetRandomSeed(problem::rand_seed);
}

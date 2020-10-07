#include <new>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>

#ifndef WIN32
#include <unistd.h>
#endif

#include <AMReX_CArena.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_AmrLevel.H>

#ifdef HYPRE
#include "_hypre_utilities.h"
#endif

#include <time.h>

using namespace amrex;

int main(int argc, char* argv[])
{
    int amrex_init_argc = argc-1;
    char* input_file_A = argv[1];
    char* input_file_B = argv[2];

    auto setup_amrex_inputs = [&](char* inputs)
    {
        int amrex_empty_argc = 0;
        char** amrex_init_argv = nullptr;

        // Clear ParmParse table
        amrex::ParmParse::Finalize();
        amrex::ParmParse::Initialize(amrex_empty_argc, amrex_init_argv, inputs);

        // Reset global AMReX Geometry defaults from ParmParse
        auto globalGeom = AMReX::top()->getDefaultGeometry();
        Geometry::Setup(nullptr, -1, nullptr, true);
    };

    amrex::Initialize(amrex_init_argc, argv);
    {
        // Initialize B (we have to do this first to initialize the BCs we want)
        setup_amrex_inputs(input_file_B);

        Amr* amrB = new Amr;

        /*
        amrB->init(0.0, 1.0);

        // Initialize A
        setup_amrex_inputs(input_file_A);

        Amr* amrA = new Amr;

        amrA->init(0.0, 1.0);

        // Load B ParmParse again in case we need it in amrB
        setup_amrex_inputs(input_file_B);

        // Transmogrify A -> B level by level
        for (int lev = 0; lev <= amrB->finestLevel(); ++lev)
        {
            // transmogrify this level's data
        }

        amrA->writePlotFile();

        // Write checkpoint file from B
        // amrB->checkPoint();

        // Write plotfiles from B
        // amrB->writePlotFile();
        // amrB->writeSmallPlotFile();
        */
    }
    amrex::Finalize();

    return 0;
}
